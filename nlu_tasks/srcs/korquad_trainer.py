import os
import sys
import timeit
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SequentialSampler, RandomSampler, DataLoader, ConcatDataset
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate
sys.path.append(os.getcwd())
from pretraining.srcs.functions import init_random, float_separator, BAR_FORMAT
from nlu_tasks.data_configs.KorQuAD.evaluate_v1_0 import eval_during_train
from nlu_tasks.srcs.nlu_utils import get_optimizer, get_lr_scheduler, get_bert_tokenizer, get_task_model
from nlu_tasks.srcs.custom_squad import SquadResult, SquadV1Processor, SquadV2Processor, squad_convert_examples_to_features


def to_list(tensor):
    return tensor.detach().cpu().tolist()

    
def train(args, train_dataset, model, tokenizer, logger, tb_writer, device):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    
    args.total_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    args.num_warmup_steps = args.warmup_ratio * args.total_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = get_optimizer(args, model)
    scheduler = get_lr_scheduler(args, optimizer)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.max_epochs}")
    logger.info(f"  Train batch size per GPU = {args.batch_size}")
    logger.info(f"  Total train batch size = {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    global_step = 1
    logger.info("  Starting fine-tuning.")
    
    train_loss = 0.0
    model.zero_grad()
    
    init_random(args.random_seed)
    running_loss = 0.
    for epoch in range(args.max_epochs):
        logger.info(f"[ {epoch+1} Epoch ]")
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training... ", bar_format=BAR_FORMAT)):
            model.train()
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            train_loss += loss.item()
            running_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if global_step % args.tb_interval == 0:
                    running_loss /= args.tb_interval
                    # logger.info(f"Epoch {epoch}, Step {step + 1}\t| Loss {running_loss:.4f} lr {scheduler.get_last_lr()}")
                    tb_writer.add_scalar(f'{args.task_name}/train_loss/step', running_loss, global_step)
                    tb_writer.add_scalar(f'{args.task_name}/train_lr/step', optimizer.param_groups[0]['lr'], global_step)
                    running_loss = 0

        logger.info(f"***** EVAL RESULTS #EP {epoch+1} *****")
        _ = evaluation(args, model, tokenizer, logger, device, epoch=epoch)
        print('\n')
    return global_step, train_loss / global_step


def evaluation(args, model, tokenizer, logger, device, epoch=None):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, logger, evaluate=True)

    os.makedirs(args.log_dir, exist_ok=True)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)

    # # Eval!
    # if logger:
    #     logger.info(f"***** Running evaluation {epoch + 1} EP *****")
    #     logger.info(f"  Num examples = {len(dataset)}")

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Dev Evaluation...", bar_format=BAR_FORMAT):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]
            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    if logger:
        logger.info(f"  Evaluation done in total {evalTime:.2f} secs ({evalTime / len(dataset):.6f} sec per example)")

    # Compute predictions
    output_prediction_file = os.path.join(args.log_dir, f"predictions_{epoch+1}ep.json")
    output_nbest_file = os.path.join(args.log_dir, f"nbest_predictions_{epoch+1}ep.json")

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.log_dir, f"null_odds_{epoch+1}ep.json")
    else:
        output_null_log_odds_file = None
    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_len,
        False,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    # Write the result
    # Write the evaluation result on file
    # if not epoch: epoch = ""
    output_eval_file = os.path.join(args.log_dir, f"eval_result.txt")

    if logger:
        logger.info("***** Official Eval Results *****")

    write_mode = 'w' if epoch == 0 else 'a'
    with open(output_eval_file, write_mode, encoding="utf-8") as f:
        official_eval_results = eval_during_train(dataset_file=f"datasets/nlu_tasks/KorQuAD/KorQuAD_v1.0_dev.json",
                                                  prediction_file=os.path.join(args.log_dir, f"predictions_{epoch+1}ep.json"))
        for key in sorted(official_eval_results.keys()):
            if key in ['official_exact_match', 'official_f1'] and logger:
                logger.info(f" {key} = {official_eval_results[key]:.2f}")
            f.write(f" [{epoch+1} Epoch]")
            f.write(f" {key} = {official_eval_results[key]}\n")

        for key in sorted(results.keys()):
            """
            key = {'HasAns_exact', 'HasAns_f1', 'HasAns_total', 
                   'best_exact', 'best_exact_thresh', 'best_f1', 'best_f1_thresh', 'exact', 'f1', 'total'}
            """
            f.write(f" {key} = {results[key]}\n")
        f.write('\n')
    return results


def load_and_cache_examples(args, tokenizer, logger, evaluate=False):
    """
    Korean Question Answering Dataset(KorQuAD) v1.0
    [https://korquad.github.io/KorQuad%201.0/]
    A large-scale Korean dataset for machine reading comprehension task
    consisting of 70,000+ human generated questions for Wikipedia articles.

    When the passage is given, the model finds the answer of the question in the phrase. The answer must be a phrase with in the passage

    {  'version': str,
       'data': [{'paragraphs': [{'qas': [{'answers': [{'text': str,
                                                       'answer_start: int}],
                                          'id': str
                                          'question': str
                                         },

                                         {'answers': [{'text': str,
                                                       'answer_start: int}],
                                          'id': str
                                          'question': str
                                         },

                                         ...
                                 'context': str
                                },
                                ...
                               ],
                 'title': str
                },
                ...
               ]
    }
    """
    doc_stride = args.doc_stride
    if tokenizer.tok_config_name in ["bts_units_var_info", "jamo_var_info"]:
        try:
            assert args.doc_stride % tokenizer.trunc_num == 0
        except AssertionError:
            doc_stride = (args.doc_stride +
                          (tokenizer.trunc_num - (args.doc_stride % tokenizer.trunc_num)))

    data_dir = f"datasets/nlu_tasks/{args.task_name}"

    tok_save_dir = os.path.join(data_dir, f"{args.tok_name}_{args.max_seq_len}t_{args.doc_stride}d_{args.max_query_len}q_{args.max_answer_len}a")
    os.makedirs(tok_save_dir, exist_ok=True)
    data_file = os.path.join(tok_save_dir, f"cached_{'dev' if evaluate else 'train'}_{args.max_seq_len}")

    if evaluate:
        if os.path.exists(data_file):
            if logger:
                logger.info(f"Loading features from cached file {data_file}")
            features_and_dataset = torch.load(data_file)
            features, dataset, examples = (
                features_and_dataset["features"],
                features_and_dataset["dataset"],
                features_and_dataset["examples"],
            )
        else:
            if logger:
                logger.info(f"Creating features from dataset file at {data_dir}")

            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            examples = processor.get_dev_examples(data_dir, 'KorQuAD_v1.0_dev.json') if evaluate \
                else processor.get_train_examples(data_dir, 'KorQuAD_v1.0_train.json')

            # For MeCab tokenizer, we remove '\n' in all texts in all examples
            for example in examples:
                example.question_text = example.question_text.replace("\n", "")
                example.context_text = example.context_text.replace("\n", "")
                if example.answer_text is not None:
                    example.answer_text = example.answer_text.replace("\n", "")
                example.title = example.title.replace("\n", "")

            features, dataset = squad_convert_examples_to_features(
                examples=examples,
                # examples=examples[:1],
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_len,
                doc_stride=doc_stride,
                max_query_length=args.max_query_len,
                is_training=not evaluate,
                return_dataset="pt",
                threads=args.threads,
            )
            if logger:
                logger.info(f"Saving features into cached file {data_file}")
            torch.save({"features": features, "dataset": dataset, "examples": examples}, data_file)
        return dataset, examples, features
    else:
        seg_num = 2
        total_dataset = []
        for i in range(seg_num):
            seg_file = data_file + f"_s{i}"

            # Init features and dataset from cache if it exists
            if os.path.exists(seg_file):
                if logger:
                    logger.info(f"Loading features from cached file {seg_file}")
                features_and_dataset = torch.load(seg_file)
                dataset = features_and_dataset["dataset"]
            else:
                if logger:
                    logger.info(f"Creating features from dataset file at {data_dir}")

                processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
                examples = processor.get_dev_examples(data_dir, 'KorQuAD_v1.0_dev.json') if evaluate \
                    else processor.get_train_examples(data_dir, 'KorQuAD_v1.0_train.json')

                seg_len = len(examples) // seg_num + 1
                segment = examples[i * seg_len: (i + 1) * seg_len]

                # For MeCab tokenizer, we remove '\n' in all texts in all examples
                for example in segment:
                    example.question_text = example.question_text.replace("\n", "")
                    example.context_text = example.context_text.replace("\n", "")
                    if example.answer_text is not None:
                        example.answer_text = example.answer_text.replace("\n", "")
                    example.title = example.title.replace("\n", "")

                logger.info(f"Creating the dataset from segment{i}: {len(segment)} examples")
                features, dataset = squad_convert_examples_to_features(
                    examples=segment,
                    # examples=examples[:1],
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_len,
                    doc_stride=doc_stride,
                    max_query_length=args.max_query_len,
                    is_training=not evaluate,
                    return_dataset="pt",
                    threads=args.threads,
                )
                if logger:
                    logger.info(f"Saving features into cached file {seg_file}")
                torch.save({"dataset": dataset}, seg_file)
                del features
                del segment
            total_dataset.append(dataset)
            del dataset
        torch.cuda.empty_cache()

        dataset = ConcatDataset(total_dataset)
        return dataset


def fine_tuning(args, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tb_writer = SummaryWriter(args.tb_dir)

    logger.info("========== fine_tuning ==========")
    logger.info(f"task name             : {args.task_name}")
    logger.info(f"tokenizer             : {args.tok_name}")
    logger.info(f"vocab size            : {args.tok_vocab_size}")
    logger.info(f"device                : {device}")
    logger.info(f"random seed           : {args.random_seed}")
    logger.info(f"total epochs          : {args.max_epochs}")
    logger.info(f"batch size            : {args.batch_size}")
    logger.info(f"accumulation_steps    : {args.gradient_accumulation_steps}")
    logger.info(f"learning rate         : {args.learning_rate}")
    logger.info(f"dropout rate          : {args.dropout_rate}")
    logger.info(f"warmup ratio          : {args.warmup_ratio}")
    logger.info(f"max_seq_len           : {args.max_seq_len}")
    logger.info(f"max_query_len         : {args.max_query_len}")
    logger.info(f"max_answer_len        : {args.max_answer_len}")
    logger.info(f"doc_stride            : {args.doc_stride}")


    #####################################
    #      Get Tokenizer and Model
    #####################################
    tokenizer = get_bert_tokenizer(args)
    _, model, _ = get_task_model(args, tokenizer)
    model.to(device)

    ################################
    #          Fine-Tuning
    ################################
    train_dataset = load_and_cache_examples(args, tokenizer, logger, evaluate=False)
    """
    # train_sampler = RandomSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    # for batch in train_dataloader:
    #     inputs = {
    #         "input_ids": batch[0],
    #         "attention_mask": batch[1],
    #         "token_type_ids": batch[2],
    #         "start_positions": batch[3],
    #         "end_positions": batch[4],
    #     }
    #     for i in range(len(batch[0])):
    #         print(f"inputs['input_ids'][{i}]: \n{inputs['input_ids'][i]}\n\n")
    #         print(f"tokenizer.decode(inputs['input_ids'][{i}]): \n{tokenizer.decode(inputs['input_ids'][i])}\n\n")
    #         print(f"inputs['start_positions'][{i}]: \n{inputs['start_positions'][i]}\n\n")
    #         print(f"inputs['end_positions'][{i}]: \n{inputs['end_positions'][i]}\n\n")
    #         print(f"answers[{i}]: \n{tokenizer.decode(inputs['input_ids'][i][inputs['start_positions'][i]: inputs['end_positions'][i]+1])}\n\n")
    # sys.exit(-111)
    """

    global_step, train_loss = train(args, train_dataset, model, tokenizer, logger, tb_writer, device)
    logger.info(f" global_step = {float_separator(global_step)}, average loss = {train_loss:.4f}")
