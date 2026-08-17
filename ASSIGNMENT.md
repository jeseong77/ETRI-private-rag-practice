# 실습 과제 — Ollama와 MCP를 연결한 내부 규정 RAG

## 목표

내부 규정 30개를 검색 가능한 색인으로 만든다. Python MCP Host에서 Ollama의 MiniMax
모델과 로컬 RAG MCP Server를 연결하고, 자연어 질문에 대한 근거와 답변을 확인한다.

## 완료 기준

1. `python index_documents.py`가 규정 30개를 색인한다.
2. 색인에 사용한 Embedding 방식이 터미널과 JSON에 기록된다.
3. `python rag_chat.py`가 RAG MCP Server를 자동으로 실행한다.
4. MCP Server가 `search_internal_rules` Tool을 공개한다.
5. MiniMax의 Tool 호출이 MCP Tool 실행과 연결된다.
6. Ollama 연결 실패 시 검색 결과와 출처가 대신 표시된다.

## 관찰할 내용

- 원본 문서의 제목 하나가 Chunk 하나로 나뉘는 과정
- 문장이 숫자 벡터로 바뀐 뒤 원문과 함께 저장되는 구조
- MCP Host가 로컬 MCP Server를 자식 프로세스로 실행하는 과정
- MiniMax가 Tool 이름·설명·입력 형식을 전달받는 과정
- Tool 실행 결과가 Ollama의 다음 입력에 추가되는 과정
- Embedding 모델, MCP Server, 대화 모델이 맡는 서로 다른 책임
