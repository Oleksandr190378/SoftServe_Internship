# Docker Test Data Directory

This directory is used for Docker testing to avoid affecting production data.

## Purpose

- **Isolated testing:** Test Docker container without modifying real data
- **Safe experimentation:** Try different configurations
- **Clean state:** Easy to reset by deleting this folder

## Structure

```
docker_test_data/
├── chroma_db/          # ChromaDB test database (created by container)
├── processed/          # Processed documents (created by container)
├── raw/               # Raw input files (copy sample files here)
└── README.md          # This file
```

## Usage

### Option 1: Start with Empty Database
```bash
# Container will create empty ChromaDB
docker-compose -f docker-compose.test.yml up -d
```

### Option 2: Copy Sample Data
```bash
# Copy some sample documents for testing
Copy-Item data/raw/papers/*.pdf docker_test_data/raw/papers/ -Recurse

# Run pipeline inside container to process
docker-compose -f docker-compose.test.yml exec streamlit-test python run_pipeline.py
```

### Option 3: Copy Existing Database
```bash
# Copy entire ChromaDB for testing (WARNING: may be large)
Copy-Item data/chroma_db docker_test_data/ -Recurse
```

## Reset Test Environment

```bash
# Stop container
docker-compose -f docker-compose.test.yml down

# Delete test data
Remove-Item docker_test_data/* -Recurse -Force

# Start fresh
docker-compose -f docker-compose.test.yml up -d
```

## Production Data Safety

✅ This directory is SEPARATE from `./data/`
✅ Production data in `./data/` is NOT affected by Docker tests
✅ Test container uses port 8502 (production uses 8501)

*python run_pipeline.py process --doc-id  realpython_python-ai-neural-network realpython_python-keras-text-classification realpython_pytorch-vs-tensorflow

*python run_pipeline.py process --doc-id   arxiv_1907_11692 arxiv_2005_11401  arxiv_1905_11946 arxiv_1906_08237

*python run_pipeline.py process --doc-id  arxiv_1608_06993 arxiv_1609_02907 arxiv_1611_05431 arxiv_1704_04861 arxiv_1707_06347   

python run_pipeline.py process --doc-id realpython_logistic-regression-python  medium_map-mrr-search-ranking medium_production-llms-nemo medium_running-evals-rag-pipeline medium_transformers-text-excel medium_vibe-proving-llms arxiv_1409_3215 arxiv_1409_4842 arxiv_1411_1784 arxiv_1502_03167 arxiv_1505_04597  

*arxiv_1703_06870 arxiv_1506_02640 arxiv_1512_03385 arxiv_1607_06450 realpython_image-processing-pillow arxiv_2001_08361


Універсальний 5-етапний план рефакторингу Python файлів:

📋 ЕТАП 1: Fix Critical Bugs & Validation
Мета: Виправити логічні помилки, які спотворюють результати

Що шукати:

❌ Неправильні умови (edge cases: порожні списки, None, zero division)
❌ Логічні помилки в обчисленнях (metrics завжди 1.0/0.0)
❌ Відсутня валідація вхідних даних
❌ Некоректна обробка порожніх колекцій
Приклад фіксів:
# ❌ BEFORE: Image hit rate завжди 1.0
if expected_images > 0:
    return len(retrieved_images) > 0  # Wrong: bool → 1.0

# ✅ AFTER: Правильний recall
if expected_images > 0:
    return len(set(retrieved) & set(expected)) / len(expected)
 📋 ЕТАП 2: Exception Handling & Constants
Мета: Зробити код стійким до помилок та конфігурабельним

Що шукати:

❌ File I/O без try-except (read/write files)
❌ API calls без error handling
❌ Hard-coded magic numbers (0.7, 0.5, 10)
❌ Hard-coded paths ("data/results.json")
Що робити:
# ❌ BEFORE: Magic numbers
if recall > 0.7 and mrr > 0.5:
    k_text = 10

# ✅ AFTER: Named constants
TARGET_RECALL = 0.7
TARGET_MRR = 0.5
DEFAULT_K_TEXT = 10

if recall > TARGET_RECALL and mrr > TARGET_MRR:
    k_text = DEFAULT_K_TEXT
File I/O pattern:
  try:
    with open(path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {path}")
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON: {e}")
     
ЕТАП 3: SOLID Principles (SRP, DRY, KISS)
Мета: Спростити код, видалити дублювання

Single Responsibility Principle:

# ❌ BEFORE: Один метод робить 5 речей
def evaluate_query(query):
    # 1. Retrieval
    chunks = retriever.retrieve(query)
    # 2. Extract IDs
    doc_ids = [c.metadata['doc_id'] for c in chunks]
    # 3. Compute metrics
    recall = calc_recall(doc_ids, relevant)
    # 4. Log results
    print(f"Recall: {recall}")
    # 5. Return metrics
    return {'recall': recall}

# ✅ AFTER: Розбити на окремі методи
def evaluate_query(query):
    chunks = self._perform_retrieval(query)
    doc_ids = self._extract_ids(chunks)
    metrics = self._compute_metrics(doc_ids)
    self._log_results(metrics)
    return metrics

 Don't Repeat Yourself:
 # ❌ BEFORE: Дублювання коду
avg_recall = sum(recalls) / len(recalls)
min_recall = min(recalls)
max_recall = max(recalls)

avg_precision = sum(precisions) / len(precisions)
min_precision = min(precisions)
max_precision = max(precisions)

# ✅ AFTER: DRY helper
def _aggregate_metric(values):
    return {
        'avg': sum(values) / len(values),
        'min': min(values),
        'max': max(values)
    }

recall_stats = _aggregate_metric(recalls)
precision_stats = _aggregate_metric(precisions)

Keep It Simple, Stupid:

Розбити складні функції на прості
Уникати вкладених циклів >2 рівнів
Переписати заплутану логіку
📋 ЕТАП 4: Dataclasses for Type Safety
Мета: Замінити Dict/Tuple на типізовані структури

Коли використовувати dataclass:
✅ Метрики/результати з багатьма полями
✅ Конфігурація з параметрами
✅ Структурні дані для JSON serialization
❌ Прості key-value пари (достатньо Dict)
Pattern:
# ❌ BEFORE: Dict hell
result = {
    'recall': 0.85,
    'precision': 0.72,
    'mrr': 0.64,
    'query_id': 1,
    'query': "what is CNN"
}

# ✅ AFTER: Type-safe dataclass
@dataclass
class QueryMetrics:
    query_id: int
    query: str
    recall: float
    precision: float
    mrr: float
    
    def to_dict(self) -> dict:
        return asdict(self)
 ЕТАП 5: Dependency Injection & Configurability
Мета: Зробити компоненти замінними та тестованими

Pattern:

# ❌ BEFORE: Hard-coded dependencies
class Evaluator:
    def __init__(self):
        self.retriever = MultimodalRetriever()  # Hard-coded
        self.output_dir = "results/"            # Hard-coded

# ✅ AFTER: Dependency Injection
class Evaluator:
    def __init__(
        self, 
        retriever: MultimodalRetriever = None,
        output_dir: str = DEFAULT_OUTPUT_DIR
    ):
        self.retriever = retriever or MultimodalRetriever()
        self.output_dir = Path(output_dir)
 БОНУС: Rounding & Formatting
Мета: Консистентність виведення
recall = round(recall, 2)
precision = round(precision, 2)
□ ЕТАП 1: Critical Bugs
  □ Edge cases (empty lists, None, zero division)
  □ Логічні помилки в обчисленнях
  □ Валідація вхідних даних

□ ЕТАП 2: Exception Handling
  □ Try-catch для File I/O
  □ Try-catch для API calls
  □ Magic numbers → Constants
  □ Hard-coded paths → Configurable

□ ЕТАП 3: SOLID
  □ SRP: Розбити великі функції
  □ DRY: Видалити дублювання
  □ KISS: Спростити складну логіку

□ ЕТАП 4: Dataclasses
  □ Metrics → @dataclass
  □ Config → @dataclass
  □ Results → @dataclass

□ ЕТАП 5: Dependency Injection
  □ Configurable paths
  □ Injectable dependencies
  □ Default values

□ БОНУС: Formatting
  □ Rounding до 2-3 знаків
  □ Консистентне виведення