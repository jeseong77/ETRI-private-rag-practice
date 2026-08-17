# ETRI 실습 — Claude Code와 Local RAG MCP Server

이 저장소는 Markdown 내부 규정을 검색 가능한 색인으로 만들고, Claude Code에서 로컬
RAG 검색 Tool을 사용하는 실습 프로젝트다.

```text
Markdown 규정
→ Chunk와 해싱 벡터 생성
→ JSON 색인 저장
→ Claude Code가 로컬 MCP Server 연결
→ search_internal_rules Tool 호출
→ 검색된 규정을 근거로 답변
```

Claude Code가 MCP Host와 대화 모델 역할을 맡는다. 수강생은 별도의 모델 실행기나
임베딩 모델을 설치하지 않는다.

## 준비물

- Python 3.10 이상
- 교육용 Claude Code Enterprise 계정
- Claude Code 설치와 로그인

## 1. 저장소 내려받기

```bash
git clone https://github.com/jeseong77/ETRI-private-rag-practice.git
cd ETRI-private-rag-practice
```

## 2. Python 환경 준비

macOS와 Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`mcp` 패키지를 내려받아야 하므로 폐쇄망에 들어가기 전에 이 단계를 완료한다.

## 3. Markdown 문서 색인

```bash
python index_documents.py
```

프로그램은 다음 순서로 실행된다.

1. `documents` 폴더의 Markdown 문서를 읽는다.
2. 두 번째 단계 제목을 기준으로 규정 30개를 Chunk 30개로 나눈다.
3. 각 Chunk를 384차원 해싱 벡터로 변환한다.
4. 벡터, 원문, 규정 제목, 파일 이름을 `storage/index.json`에 저장한다.

해싱 벡터는 단어와 글자가 겹치는 정도를 숫자로 표현하는 교육용 구현이다. 실제 의미
임베딩 모델과 동일하지 않지만 문서 분할, 벡터 저장, 유사도 검색, RAG Tool 연결 과정을
인터넷 없이 실습할 수 있다.

## 4. Local MCP Server 등록

가상 환경이 활성화된 같은 터미널에서 실행한다.

```bash
python configure_claude_mcp.py
```

설정 프로그램은 현재 가상 환경의 Python 전체 경로와 `rag_mcp_server.py` 전체 경로를
Claude Code의 현재 프로젝트 전용 설정에 등록한다. macOS의 `python3`와 Windows의
`python` 명령 차이를 사용자가 직접 처리할 필요가 없다.

터미널 결과에서 다음 상태를 확인한다.

```text
internal-rules-rag
Scope: Local config
Status: Connected
```

## 5. Claude Code 실행

가상 환경이 활성화된 같은 터미널에서 실행한다.

```bash
claude
```

Claude Code 안에서 다음 명령으로 연결 상태를 확인한다.

```text
/mcp
```

목록에 다음 항목이 나타나야 한다.

```text
internal-rules-rag
Tool: search_internal_rules
```

## 6. 규정 질문

Claude Code에 다음과 같이 요청한다.

```text
내부 규정 검색 도구를 사용해서 USB를 외부로 반출하려면
누구의 승인이 필요한지 알려줘. 사용한 규정과 파일도 표시해줘.
```

내부에서는 다음 일이 발생한다.

1. Claude Code가 `search_internal_rules` Tool의 이름·설명·입력 형식을 확인한다.
2. Claude가 질문과 `top_k` 값을 Tool 인자로 만든다.
3. Claude Code의 MCP Client가 로컬 MCP Server에 `tools/call` 요청을 보낸다.
4. MCP Server가 `storage/index.json`에서 질문과 가까운 규정을 찾는다.
5. MCP Server가 규정 제목, 원문, 출처, 유사도를 반환한다.
6. Claude가 반환된 원문만 근거로 답변하고 출처를 표시한다.

`CLAUDE.md`에는 내부 규정 질문에서 Tool을 먼저 사용하고, 검색되지 않은 내용을 추측하지
않도록 하는 프로젝트 지침이 들어 있다.

## MCP Tool

```text
이름: search_internal_rules

입력:
- query: 검색할 자연어 질문
- top_k: 반환할 규정 수, 기본값 3, 최대 5

출력:
- 검색 순위와 유사도
- 규정 제목과 원문
- 원본 Markdown 파일 이름
- 색인에 사용한 벡터 방식
```

## 파일 구성

```text
documents/                         가상의 내부 규정 30개
storage/index.json                 생성된 원문·벡터 색인
index_documents.py                 Markdown 색인 프로그램
rag_mcp_server.py                  로컬 RAG MCP Server
configure_claude_mcp.py            현재 컴퓨터에 맞는 MCP 설정 등록
CLAUDE.md                          검색 Tool 사용 지침
rag_practice/chunking.py           Markdown을 Chunk로 분리
rag_practice/embedding.py          해싱 벡터 생성
rag_practice/retrieval.py          코사인 유사도 검색
```

## 문제가 생겼을 때

- `storage/index.json`이 없으면 `python index_documents.py`를 먼저 실행한다.
- MCP Server가 연결되지 않으면 `.venv` 폴더와 등록 명령 실행 여부를 확인한다.
- Python 패키지를 찾지 못하면 `python -m pip install -r requirements.txt`를 다시 실행한다.
- 등록 상태는 `claude mcp get internal-rules-rag`로 확인한다.

## 검증

```bash
python -m unittest discover -s tests -v
```

검증에는 다음 항목이 포함된다.

- Markdown 규정 30개 Chunk 분리
- 질문과 관련된 USB 반출 규정 검색
- 현재 가상 환경 Python을 사용하는 Claude Code 등록 명령
- MCP Server의 Tool 목록과 실제 검색 결과
