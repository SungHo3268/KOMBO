# Tokenization
<br/>


## 1. Jamo Tokenizer
* text: 테스트 문장입니다.
* tokens: ['ᄐ', 'ᅦ', 'ᄉ', 'ᅳ', 'ᄐ', 'ᅳ', '▁', 'ᄆ', 'ᅮ', 'ᆫ', 'ᄌ', 'ᅡ', 'ᆼ', 'ᄋ', 'ᅵ', 'ᆸ', 'ᄂ', 'ᅵ', 'ᄃ', 'ᅡ', '.']
* detok: 테스트 문장입니다.
<br/>

## 2. Character Tokenizer
* text: 테스트 문장입니다.
* tokens: ['테', '스', '트', '▁', '문', '장', '입', '니', '다', '.']
* detok: 테스트 문장입니다.
<br/>

## 3. Morpheme Tokenizer
* text: 테스트 문장입니다.
* tokens: ['테스트', '▃', '문장', '입니다', '.']
* detok: 테스트 문장입니다.
<br/>

## 4. Subword(SentencePiece) Tokenizer
#### Vocab size: 4K
* text: 테스트 문장입니다.
* tokens: ['▁테', '스트', '▁문', '장', '입', '니', '다', '.']
* detok: 테스트 문장입니다.
#### Vocab size: 8K
* text: 테스트 문장입니다.
* tokens: ['▁테', '스트', '▁문', '장', '입', '니다', '.']
* detok: 테스트 문장입니다.
#### Vocab size: 16K
* text: 테스트 문장입니다.
* tokens: ['▁테스트', '▁문장', '입니다', '.']
* detok: 테스트 문장입니다.
#### Vocab size: 32K
* text: 테스트 문장입니다.
* tokens: ['▁테스트', '▁문장', '입니다', '.']
* detok: 테스트 문장입니다.
#### Vocab size: 64K
* text: 테스트 문장입니다.
* tokens: ['▁테스트', '▁문장', '입니다', '.']
* detok: 테스트 문장입니다.
<br/>

## 5. Morpheme-aware Subword Tokenizer
#### Vocab size: 4K
* text: 테스트 문장입니다.
* tokens: ['▁테', '스트', '▃', '▁문', '장', '▁입', '니', '다', '▁.']
* detok: 테스트 문장입니다.
#### Vocab size: 8K
* text: 테스트 문장입니다.
* tokens: ['▁테스트', '▃', '▁문장', '▁입니다', '▁.']
* detok: 테스트 문장입니다.
#### Vocab size: 16K
* text: 테스트 문장입니다.
* tokens: ['▁테스트', '▃', '▁문장', '▁입니다', '▁.']
* detok: 테스트 문장입니다.
#### Vocab size: 32K
* text: 테스트 문장입니다.
* tokens: ['▁테스트', '▃', '▁문장', '▁입니다', '▁.']
* detok: 테스트 문장입니다.
#### Vocab size: 64K
* text: 테스트 문장입니다.
* tokens: ['▁테스트', '▃', '▁문장', '▁입니다', '▁.']
* detok: 테스트 문장입니다.
<br/>

## 6. Word Tokenizer
<div>
    <code><B>[Notice]</B></code> Moses Tokenizer 가 한글을 제대로 지원 하지 않는다. 따라서 토큰들을 띄어쓰기 기준으로 detokenizing 했기 때문에 마지막 구두점이 컨트롤 되지 않는다.
    문장 끝 구두점 앞의 여백 관리 코드를 추가하여 주석처리 해 놓았으니, 해당 코드의 주석을 해제하여 사용하면 된다.
</div><br/>

* text: 테스트 문장입니다.
* tokens: ['테스트', '문장입니다', '.']
* detok: 테스트 문장입니다 .
<br/>

