# ETRI 실습 — Local Embedding과 Private RAG

이 저장소는 내부 문서를 작은 조각으로 나누고, 각 조각을 벡터로 변환한 뒤,
질문과 가까운 원문을 찾는 과정을 터미널에서 확인하는 실습 프로젝트다.

```text
문서 읽기 → Chunk 생성 → Embedding 생성 → 검색 색인 저장
질문 입력 → 질문 Embedding 생성 → 유사한 Chunk 검색 → 답변용 근거 구성
```

## 준비물

- Python 3.9 이상
- Ollama: EmbeddingGemma를 사용할 때만 필요하다.

파이썬 외부 패키지는 사용하지 않는다. Ollama가 없거나 임베딩 모델을 내려받지 못해도
해싱 방식의 대체 벡터로 전체 실습을 끝낼 수 있다.

## 가장 짧은 실행 방법

```bash
python main.py
```

`main.py`는 Ollama와 EmbeddingGemma를 확인하고, 모델이 없으면 자동으로 다운로드를
시도한다. Ollama가 없거나 폐쇄망에서 다운로드가 실패하면 프로그램을 종료하지 않고
해싱 대체 모드로 문서 분리, 색인, 검색, RAG 입력 생성을 이어서 실행한다.

`requirements.txt`는 파이썬 패키지만 설치할 수 있다. Ollama는 모델을 실행하는 별도
프로그램이므로 `requirements.txt`로 설치할 수 없다. 이 저장소는 외부 파이썬 패키지를
사용하지 않아 패키지 설치 단계 자체가 필요하지 않다.

## 1. 저장소 내려받기

```bash
git clone https://github.com/jeseong77/ETRI-private-rag-practice.git
cd ETRI-private-rag-practice
```

## 2. 임베딩 모델 준비

```bash
python prepare.py
```

이 명령은 다음 순서로 동작한다.

1. Ollama 설치 여부를 확인한다.
2. 로컬 Ollama가 실행 중인지 확인한다.
3. `embeddinggemma:300m-qat-q4_0` 모델이 없으면 내려받는다.
4. 준비에 실패해도 해싱 대체 모드로 실습할 수 있음을 안내한다.

EmbeddingGemma 압축 모델은 약 239MB다. 한 번 내려받으면 이후 임베딩 계산은
로컬 컴퓨터에서 수행된다.

## 3. 문서 색인

```bash
python ingest.py
```

`documents/보안_및_자료_취급_규정.md`에 있는 규정 30개를 읽어 30개의 Chunk로
나누고, 각 Chunk의 벡터와 원문을 `storage/index.json`에 저장한다.

Ollama 또는 모델을 사용할 수 없으면 자동으로 해싱 대체 모드가 선택된다. 이 모드는
단어가 겹치는 정도를 벡터로 표현하는 교육용 대체 구현이며 의미를 이해하는 임베딩
모델과 성능이 같지 않다. 현재 사용 중인 방식은 터미널과 색인 파일에 기록된다.

대체 모드를 직접 선택하려면 다음 명령을 사용한다.

```bash
python ingest.py --backend hash
```

## 4. 유사한 규정 검색

```bash
python search.py "USB를 외부로 반출하려면 어떤 승인이 필요한가?"
```

질문을 현재 색인과 같은 방식으로 벡터화하고, 코사인 유사도가 높은 규정부터
보여 준다. 코사인 유사도는 두 벡터의 방향이 얼마나 가까운지를 나타내는 값이다.

## 5. RAG 입력 확인

```bash
python ask.py "USB를 외부로 반출하려면 어떤 승인이 필요한가?"
```

검색된 원문을 LLM에 전달할 수 있는 프롬프트로 조립해 보여 준다. 여기까지가
Retrieval-Augmented Generation에서 Retrieval, 즉 근거 검색과 입력 보강에 해당한다.

Ollama에 대화 모델이 이미 설치되어 있다면 모델 이름을 지정해 답변 생성까지 실행할
수 있다.

```bash
python ask.py "USB를 외부로 반출하려면 어떤 승인이 필요한가?" --chat-model gemma4:e2b
```

대화 모델은 이 저장소가 자동으로 내려받지 않는다. 수강 환경의 컴퓨터 성능과 이미
설치된 모델이 다를 수 있기 때문이다.

## 색인을 다시 만들어야 하는 경우

- 원본 문서를 수정한 경우
- Chunk 분리 기준을 바꾼 경우
- 임베딩 모델을 바꾼 경우
- 해싱 대체 모드에서 EmbeddingGemma 방식으로 전환한 경우

서로 다른 모델이 만든 벡터는 같은 공간에서 비교할 수 없다. 색인을 만들 때 사용한
방식은 `storage/index.json`에 저장되며 검색도 반드시 같은 방식을 사용한다.

## 파일 구성

```text
documents/                         실습용 규정 원문
rag_practice/chunking.py           Markdown 문서를 Chunk로 분리
rag_practice/embedding.py          Ollama와 해싱 임베딩 구현
rag_practice/index_store.py        벡터와 원문 저장
rag_practice/retrieval.py          코사인 유사도 검색
rag_practice/generation.py         검색 근거와 질문을 LLM 입력으로 구성
prepare.py                         Ollama와 모델 준비 확인
main.py                            준비부터 검색까지 한 번에 실행
ingest.py                          문서 색인 생성
search.py                          관련 Chunk 검색
ask.py                             RAG 입력 확인과 선택적 답변 생성
```

## 검증

```bash
python -m unittest discover -s tests -v
```

검증은 외부 네트워크를 사용하지 않는다.
