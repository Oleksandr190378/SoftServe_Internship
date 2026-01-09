# Результати оцінки Retrieval Quality (Phase D)

**Дата:** 8 січня 2026  
**План:** [evaluation_plan.md](evaluation_plan.md) - ВАРІАНТ 1 (Швидкий старт)  
**Етап:** A2 - Базова оцінка ретриву

---

## 🎯 Фінальні результати

### Досягнуті метрики

| Метрика | Результат | Ціль | Статус | Покращення |
|---------|-----------|------|--------|------------|
| **Recall@5** | **95.0%** | ≥70% | ✅ | +25.0% |
| **Image Hit Rate** | **88.9%** | ≥60% | ✅ | +28.9% |
| **MRR** | **1.000** | ≥0.70 | ✅ | +30.0% |

**Висновок**: Всі 3 цілі Phase D досягнуті! 🎉

---

## 📊 Історія оцінки

### Перша оцінка (після виправлення ChromaDB path)

**Дата:** 8 січня 2026 13:49:56  
**Файл:** `eval/results/retrieval_eval_20260108_134956.json`

**Результати:**
- Recall@5: 95.0% ✅
- Image Hit Rate: 55.6% ❌ (нижче цілі 60%)
- MRR: 1.000 ✅
- **Проблема**: Image Recall = 0.0% для всіх queries

**Root Cause:**
```python
# debug_metadata.py показав:
"All metadata keys: ['enriched_caption', 'doc_id', 'filename', ...]"
# ❌ Поле 'image_id' ВІДСУТНЄ!
```

### Друга оцінка (після додавання image_id)

**Дата:** 8 січня 2026 (після rebuild ChromaDB)

**Результати:**
- Recall@5: 95.0% ✅
- Image Hit Rate: 88.9% ✅ (+33.3% покращення!)
- MRR: 1.000 ✅

**Що виправили:**
1. Додали `'image_id': image_id` в metadata ([index/build_index.py](../index/build_index.py#L218))
2. Виправили JSON десеріалізацію related_image_ids ([rag/retriever.py](../rag/retriever.py))
3. Перебудували ChromaDB з --force для 19 документів

---

## 🐛 Критичні баги виявлені та виправлені

### Bug 1: ChromaDB Path Mismatch

**Симптом:** `evaluate_retrieval.py` повертав 0 результатів для всіх queries

**Root Cause:**
```python
# build_index.py використовував:
chromadb.PersistentClient(path="data/chroma_db")

# retriever.py використовував:
persist_directory=str(chroma_dir / collection_name)  # ❌ Subdirectory!
```

**Рішення:**
```python
# retriever.py тепер використовує parent directory:
persist_directory=str(chroma_dir)  # ✅
```

**Результат:** Text retrieval запрацював (Recall@5=95%, MRR=1.0)

---

### Bug 2: Missing image_id in ChromaDB Metadata

**Симптом:** Image Hit Rate=55.6%, Image Recall=0.0%, `fetch_images_by_ids()` повертає 0 images

**Root Cause:**
```python
# debug_metadata.py показав:
results = image_collection.get(
    where={"image_id": "arxiv_1207_0580_embedded_001"}
)
# ❌ Повертає 0 результатів, бо metadata НЕ має поля 'image_id'!

# build_index.py lines 215-226 НЕ додавав image_id:
metadata = {
    'doc_id': img['doc_id'],
    'filename': img['filename'],
    # ❌ 'image_id' відсутній!
}
```

**Рішення:**
```python
# build_index.py line 218 тепер додає image_id:
metadata = {
    'image_id': image_id,  # ✅ CRITICAL для fetch_images_by_ids()
    'doc_id': img['doc_id'],
    'filename': img['filename'],
    ...
}
```

**Результат:** Image Hit Rate покращився з 55.6% → 88.9% (+33.3%)

---

### Bug 3: JSON Metadata Serialization

**Симптом:** `related_image_ids` зберігались як JSON string `'["img1","img2"]'` але парсились як comma-separated

**Root Cause:**
```python
# build_index.py використовує:
'related_image_ids': json.dumps(chunk['related_image_ids'])

# retriever.py очікував plain string:
ids = metadata.get('related_image_ids', '').split(',')  # ❌
```

**Рішення:**
```python
# retriever.py lines 337-355 тепер десеріалізує JSON:
ids_str = metadata.get('related_image_ids', '')
if isinstance(ids_str, str) and ids_str.startswith('['):
    related_ids = json.loads(ids_str)  # ✅
else:
    related_ids = [id.strip() for id in ids_str.split(',') if id.strip()]
```

**Результат:** Правильна обробка JSON metadata в 4 місцях retriever.py

---

## 📁 Ground Truth Dataset

**Файл:** [eval/ground_truth.json](../eval/ground_truth.json)  
**Розмір:** 10 queries, 10 documents, 16 images

### Розподіл queries:

| Тип | Кількість | Приклади |
|-----|-----------|----------|
| **Text** | 5 | "dropout regularization", "Transformer architecture" |
| **Visual** | 3 | "GAN discriminator diagram", "NumPy array visualization" |
| **Hybrid** | 2 | "agents planning tasks", "RAG chunk size" |

### Валідація:

```bash
python eval/validate_ground_truth.py
```

**Результат:** ✅ 100% успіх - всі 10 docs і 16 images існують в системі

---

## 🔧 Технічні деталі

### Evaluation Metrics

**1. Recall@k**
```python
recall = len(retrieved_relevant) / len(relevant_docs)
# k = [3, 5, 10]
```

**2. Precision@k**
```python
precision = len(retrieved_relevant) / len(retrieved_docs[:k])
```

**3. Mean Reciprocal Rank (MRR)**
```python
# Позиція першого релевантного документа
reciprocal_rank = 1.0 / rank if rank > 0 else 0
```

**4. Image Hit Rate**
```python
# % queries де хоча б 1 релевантне зображення знайдено
hit_rate = queries_with_images / total_queries_with_relevant_images
```

### Retrieval Configuration

```python
# rag/retriever.py
text_results = 5  # Top-5 text chunks
images_per_chunk = 2  # Max 2 images per chunk
rerank = True  # Reranking enabled
```


### Після виправлення:

```json
{
  "query_id": "q03",
  "query": "Show me the GAN discriminator and generator architecture",
  "retrieved_image_ids": ["arxiv_1406_2661_embedded_001"],  // ✅
  "expected_image_ids": ["arxiv_1406_2661_embedded_001"],
  "image_precision": 1.0,
  "image_recall": 1.0
}
```

**Результат:** Image Hit Rate 55.6% → 88.9%

---

## 🔍 Приклад успішного retrieval

### Query: "Explain dropout regularization technique"

**Retrieved Documents (Top-5):**
1. ✅ arxiv_1207_0580 (Dropout paper) - Rank 1
2. ✅ realpython_gradient-descent-algorithm-python - Rank 2
3. ❌ arxiv_1409_1556 (GRU) - Rank 3
4. ❌ medium_illustrated-transformer - Rank 4
5. ❌ arxiv_1706_03762 (Transformer) - Rank 5

**Retrieved Images:**
1. ✅ arxiv_1207_0580_embedded_001 (Dropout diagram)
2. ✅ arxiv_1207_0580_embedded_002 (Comparison chart)

**Metrics:**
- Recall@5: 1.0 (1/1 relevant found)
- Precision@5: 0.2 (1/5 is relevant)
- Reciprocal Rank: 1.0 (relevant at position 1)
- Image Recall: 1.0 (2/2 expected images found)

**MRR = 1.0**: Найкращий можливий результат!

---

## 🎓 Висновки

### Що працює добре:

✅ **Text retrieval**: Recall@5=95%, MRR=1.0 - відмінна якість пошуку  
✅ **Image retrieval**: Hit Rate=88.9% - знаходить релевантні зображення в 8/9 queries  
✅ **ChromaDB**: Multimodal indexing з JSON metadata працює стабільно  
✅ **Enriched captions**: VLM + author + context дають хороші embeddings

### Виявлені проблеми:

❌ **ChromaDB path compatibility**: Native client vs LangChain wrapper мали різні paths  
❌ **Metadata schema**: image_id не був в metadata спочатку  
❌ **JSON serialization**: Потрібна explicit десеріалізація для lists в metadata

### Рекомендації для Phase D.B1 (Faithfulness):

1. **LLM Judge**: Використати GPT-4o-mini для оцінки groundedness відповідей
2. **Target**: Faithfulness score ≥4.0/5.0
3. **Cost**: ~$0.10 для 10 queries (GPT-4o-mini дешевий)
4. **Metrics**: Relevance, Completeness, Accuracy, Citation quality

---

## 📚 Файли

**Evaluation Infrastructure:**
- [eval/ground_truth.json](../eval/ground_truth.json) - 10 annotated queries
- [eval/validate_ground_truth.py](../eval/validate_ground_truth.py) - Validation script
- [eval/evaluate_retrieval.py](../eval/evaluate_retrieval.py) - Evaluation system (347 lines)
- [eval/results/](../eval/results/) - JSON results з усіма runs



**Fixed Code:**
- [rag/retriever.py](../rag/retriever.py) - ChromaDB path + JSON deserialization
- [index/build_index.py](../index/build_index.py) - Added image_id to metadata



## ⏭️ Наступні кроки

**Phase D.B1: Faithfulness Judge** 
- Implement LLM-based answer evaluation
- Test on 10 ground truth queries
- Calculate faithfulness scores
- Target: ≥4.0/5.0 average

**Phase D: Final Report** (~1 година)
- Document Top 3 improvements with ROI analysis
- Cost/quality trade-offs
- Recommendations for production

**Загальний прогрес Phase D:**
- ✅ A1: Ground Truth Dataset
- ✅ A2: Retrieval Evaluation (Recall, MRR, Image Hit Rate)
- ⏳ B1: Faithfulness Judge with LLM
- ⏳ D: Final Report


