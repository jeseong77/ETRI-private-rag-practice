# ETRI 실습 — Ollama, MCP Host, Private RAG

이 저장소는 Markdown 내부 규정을 검색 가능한 색인으로 만들고, 로컬 RAG MCP Server의
검색 Tool을 Ollama 모델과 연결하는 실습 프로젝트다.

수강생이 직접 실행하는 프로그램은 두 개다.

```text
1. index_documents.py
   Markdown → Chunk → Embedding → JSON 색인

2. rag_chat.py
   사용자 질문 → Ollama → MCP Tool 호출 → RAG 검색 → Ollama 최종 답변
```

`rag_mcp_server.py`는 `rag_chat.py`가 자동으로 실행한다. 별도 터미널에서 실행하지 않는다.

## 전체 구조

```text
documents/*.md
      │
      ▼
index_documents.py ──→ storage/index.json
                              ▲
                              │ 검색
사용자 ──→ rag_chat.py ──MCP──→ rag_mcp_server.py
                │
                └──── Ollama / MiniMax M3 Cloud
```

## 준비물

- Python 3.10 이상
- Ollama
- MiniMax M3 Cloud 사용 시 인터넷 연결과 Ollama 로그인

Ollama 설치 파일은 [공식 다운로드 페이지](https://ollama.com/download)에서 받는다.

## 1. 저장소와 Python 환경 준비

```bash
git clone https://github.com/jeseong77/ETRI-private-rag-practice.git
cd ETRI-private-rag-practice
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell에서는 다음 명령으로 가상 환경을 활성화한다.

```powershell
.venv\Scripts\Activate.ps1
```

폐쇄망에서는 `mcp` 패키지를 새로 받을 수 없다. 인터넷이 연결된 장소에서 이 준비
단계를 먼저 완료해야 한다.

## 2. Markdown 문서 색인

```bash
python index_documents.py
```

프로그램은 다음 순서로 실행된다.

1. Ollama가 설치되어 있는지 확인한다.
2. Ollama가 꺼져 있으면 실행을 시도한다.
3. EmbeddingGemma 압축 모델이 없으면 다운로드를 시도한다.
4. 규정 30개를 Chunk 30개로 분리한다.
5. 각 Chunk의 Embedding과 원문을 `storage/index.json`에 저장한다.

Ollama가 없거나 폐쇄망에서 모델 다운로드가 실패하면 프로그램은 종료되지 않는다.
SHA-256 해싱 방식으로 384차원 교육용 벡터를 만들고 같은 JSON 색인을 생성한다.

해싱 방식은 단어와 글자가 겹치는 정도를 표현한다. 문장의 의미를 학습한 EmbeddingGemma와
성능이 같지는 않지만 Chunk 생성, 벡터 저장, 유사도 검색 구조를 실습할 수 있다.

해싱 방식을 직접 선택하려면 다음 명령을 사용한다.

```bash
python index_documents.py --backend hash
```

## 3. MiniMax M3 Cloud 준비

MiniMax M3는 Ollama Cloud에서 실행된다. 인터넷 연결이 필요하다.

```bash
ollama signin
ollama pull minimax-m3:cloud
```

로그인에는 개인 Ollama 계정을 사용한다. 모델이 준비되지 않았거나 인터넷이 끊기면
RAG Chat은 종료되지 않고 MCP 검색 결과 원문을 그대로 보여 준다.

## 4. RAG Chat 실행

```bash
python rag_chat.py
```

실행되면 다음 세 상태가 표시된다.

```text
[색인] 저장된 Chunk와 Embedding 방식
[Ollama] MiniMax 연결 상태
[MCP] search_internal_rules Tool 연결 상태
```

질문 예시:

```text
USB를 외부로 반출하려면 누구의 승인이 필요한가?
```

내부에서는 다음 일이 순서대로 발생한다.

1. MCP Host인 `rag_chat.py`가 MiniMax에 질문과 Tool 설명을 전달한다.
2. MiniMax가 `search_internal_rules` Tool 호출을 요청한다.
3. MCP Host가 로컬 MCP Server에 `tools/call`을 보낸다.
4. MCP Server가 `storage/index.json`에서 관련 규정을 검색한다.
5. MCP Host가 검색 결과를 Ollama 대화에 Tool 결과로 추가한다.
6. MiniMax가 검색된 원문만 근거로 최종 답변을 작성한다.

모델이 Tool을 호출하지 않으면 MCP Host가 검색 Tool을 직접 실행한 뒤 근거를 다시
전달한다. Ollama 호출 자체가 실패하면 검색 결과와 출처까지만 출력한다.

한 번 질문한 뒤 종료하려면 다음 명령을 사용한다.

```bash
python rag_chat.py --question "USB를 외부로 반출하려면 누구의 승인이 필요한가?"
```

## MCP Tool

로컬 MCP Server는 Tool 하나만 공개한다.

```text
이름: search_internal_rules

입력:
- query: 검색할 자연어 질문
- top_k: 반환할 규정 수, 기본값 3, 최대 5

출력:
- 순위와 유사도
- 규정 제목과 원문
- 원본 파일 이름
- 색인에 사용한 Embedding 방식
```

## 폐쇄망 동작

```text
EmbeddingGemma 다운로드 실패
→ 해싱 벡터 사용
→ 색인과 MCP 검색 가능

MiniMax M3 Cloud 연결 실패
→ 자연어 최종 답변 생성 불가
→ MCP 검색 결과와 출처 출력
```

Embedding 폴백과 대화 모델 폴백은 서로 다르다. 해싱 방식은 검색용 벡터만 대신하며
MiniMax처럼 답변을 작성하지는 않는다.

## 파일 구성

```text
documents/                         가상의 내부 규정 30개
storage/index.json                 생성된 원문·벡터 색인
index_documents.py                 첫 번째 실습 프로그램
rag_chat.py                        두 번째 실습 프로그램, MCP Host
rag_mcp_server.py                  로컬 RAG MCP Server
rag_practice/chunking.py           Markdown을 Chunk로 분리
rag_practice/embedding.py          Ollama와 해싱 Embedding
rag_practice/retrieval.py          코사인 유사도 검색
rag_practice/ollama_chat.py        Ollama Chat API 연결
```

## 검증

```bash
python -m unittest discover -s tests -v
```

검증에는 다음 항목이 포함된다.

- 규정 30개 Chunk 분리
- 해싱 검색 결과
- Ollama 연결 실패 폴백
- Ollama Embedding API 응답 처리
- MCP Server Tool 목록과 실행
- Ollama Tool 호출을 MCP Server 실행으로 연결하는 Host 루프
