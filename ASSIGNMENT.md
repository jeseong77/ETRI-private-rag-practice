# 실습 과제 — Claude Code와 MCP를 연결한 내부 규정 RAG

## 목표

내부 규정 30개를 검색 가능한 색인으로 만든다. Claude Code에 로컬 RAG MCP Server를
연결하고, 자연어 질문이 MCP Tool 호출과 근거 기반 답변으로 이어지는 과정을 확인한다.

## 완료 기준

1. `python index_documents.py`가 규정 30개를 색인한다.
2. `python configure_claude_mcp.py`가 현재 가상 환경으로 Server를 등록한다.
3. Claude Code의 `/mcp` 화면에 `internal-rules-rag`가 연결 상태로 나타난다.
4. MCP Server가 `search_internal_rules` Tool을 공개한다.
5. Claude가 자연어 질문을 Tool 인자로 변환한다.
6. 검색 결과와 최종 답변에 규정 제목, 원문, 출처가 포함된다.

## 관찰할 내용

- 원본 문서의 제목 하나가 Chunk 하나로 나뉘는 과정
- 문장이 숫자 벡터로 바뀐 뒤 원문과 함께 저장되는 구조
- Claude Code가 등록된 명령으로 로컬 Server를 실행하는 과정
- Claude가 Tool 이름·설명·입력 형식을 전달받는 과정
- Tool 실행 결과가 Claude의 대화 맥락에 추가되는 과정
- 색인 프로그램, MCP Server, Claude Code가 맡는 서로 다른 책임
