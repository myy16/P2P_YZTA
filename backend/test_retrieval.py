from app.core.retriever import get_retriever
r = get_retriever()
q = 'Mohammed Yusuf Yılmaz hangi yarışmaya katılmış'
print("Query tokens:", r._tokenize(q))
res = r.vector_store.query(r.embedding_service.embed_query(q), top_k=10)
for idx, text in enumerate(res['documents'][0]):
    print(f"--- Chunk {idx} ---")
    print(text)
    lex = r._lexical_overlap(r._tokenize(q), r._tokenize(text))
    dist = res['distances'][0][idx] if 'distances' in res and res['distances'] else None
    sem = r._semantic_similarity(dist)
    print(f"Lexical: {lex}, Semantic: {sem}")
