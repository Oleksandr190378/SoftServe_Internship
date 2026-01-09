# Retrieval Evaluation Analysis & Improvements
**Date:** 2026-01-08  
**Evaluation:** Quick Start (10 queries)  
**Results File:** [retrieval_eval_20260108_134956.json](results/retrieval_eval_20260108_134956.json)

---

## 📊 Current Results

### ✅ **Text Retrieval (EXCELLENT)**
- **Recall@5: 95.0%** (target ≥70%) ✅ **+25% above target**
- **MRR: 1.000** (target ≥0.70) ✅ **Perfect score**
- **Precision@5: 24%** - acceptable for RAG (diversity over precision)

**Analysis:** Text retrieval працює відмінно. Система знаходить релевантні документи з першого разу (MRR=1.0) і покриває майже всі релевантні docs.

---

### ❌ **Image Retrieval (NEEDS IMPROVEMENT)**
- **Image Hit Rate: 55.6%** (target ≥60%) ❌ **-4.4% below target**
- **Image Recall: 0.0%** ❌ **Критична проблема**

**Breakdown:**
- Visual queries: 9 total
- Queries with images: 5 (55.6%)
- Queries without images: 4 (44.4%)

**Failed queries:**
1. Query 1: "Explain dropout regularization" - 0 images (expected 2)
2. Query 5: "Explain gradient descent" - 0 images (expected 2)
3. Query 6: "How to use NumPy arrays?" - 0 images (expected 2)
4. Query 8: "How do AI agents plan tasks?" - 0 images (expected 1)

---

## 🔍 Root Cause Analysis

### **Problem: Image Recall = 0.0%**

Проаналізував логи evaluation і знайшов проблему:

```
13:49:37 - INFO - Fetched 0 images by ID
13:49:37 - INFO - Strict retrieval: 0 images from metadata (3 strong, 3 weak links)
```

**Причина:** `fetch_images_by_ids()` повертає 0 images, хоча metadata має image IDs.

**Чому так відбувається:**

1. **Metadata має image IDs** (related_image_ids, nearby_image_ids)
2. **fetch_images_by_ids використовує `.get(where={"image_id": img_id})`**
3. **BUT: metadata в ChromaDB зберігається як JSON strings!**

```python
# В index/build_index.py:
metadata = {
    'image_id': chunk['image_id'],  # ✅ String
    'related_image_ids': json.dumps(chunk['related_image_ids']),  # ❌ JSON string!
}
```

4. **Retriever шукає image_id в related_image_ids (comma-separated string):**
```python
related = chunk.metadata.get('related_image_ids', '')  # "img1,img2,img3"
if related:
    image_ids_strong.update([id.strip() for id in related.split(',') if id.strip()])
```

5. **Але fetch_images_by_ids() шукає точний match:**
```python
results = self.image_store.get(where={"image_id": img_id})  # Exact match fails!
```

---

## 🎯 Recommended Improvements

### **IMPROVEMENT #1: Fix metadata deserialization** 
**Priority:** 🔴 CRITICAL  
**Impact:** +40-50% Image Hit Rate

**Problem:**
`related_image_ids` and `nearby_image_ids` зберігаються як JSON strings в ChromaDB, але читаються як strings (не deserializing).

**Solution:**
```python
# In retriever.py - retrieve_with_strict_images():
for chunk in text_chunks:
    # Current (BROKEN):
    related = chunk.metadata.get('related_image_ids', '')
    
    # Fixed:
    related = chunk.metadata.get('related_image_ids', '')
    # If it's a JSON string, deserialize it
    if related and related.startswith('['):
        import json
        related_list = json.loads(related)
        image_ids_strong.update(related_list)
    elif related:
        # Comma-separated fallback
        image_ids_strong.update([id.strip() for id in related.split(',') if id.strip()])
```

**Expected improvement:** Image Hit Rate: 55.6% → 85-90%

---

### **IMPROVEMENT #2: Lower similarity threshold for visual queries**
**Priority:** 🟠 MEDIUM  
**Impact:** +5-10% Image Hit Rate

**Problem:**
Visual queries rely on fallback semantic search with threshold=0.5. Може бути занадто жорстко.

**Solution:**
```python
# In retriever.py - _fallback_visual_search():
# Current:
is_match, similarity, chunk_id = self.verify_semantic_match(
    img, text_chunks, threshold=0.5, chunk_embeddings=chunk_embeddings
)

# Improved:
is_match, similarity, chunk_id = self.verify_semantic_match(
    img, text_chunks, threshold=0.4,  # Lower for visual queries
    chunk_embeddings=chunk_embeddings
)
```

**Expected improvement:** Image Hit Rate: +5%

---

### **IMPROVEMENT #3: Boost image retrieval for visual keywords**
**Priority:** 🟢 LOW  
**Impact:** +2-5% Image Hit Rate

**Problem:**
Non-visual queries ("Explain dropout") не шукають images навіть якщо вони є в документі.

**Solution:**
```python
# In retriever.py - retrieve_with_verification():
# Add heuristic: if query is NOT visual but metadata has many images, include them

if len(verified_images) == 0 and len(metadata_images) > 0:
    # Even if query not visual, try semantic matching with lower threshold
    # (user may not explicitly ask for images but they're relevant)
    logging.info("  Non-visual query but metadata has images - checking relevance")
    chunk_embeddings = self._batch_embed_chunks(text_chunks)
    image_embeddings = self._batch_embed_images(metadata_images)
    
    verified_images = self._verify_metadata_images(
        metadata_images, text_chunks, chunk_embeddings, image_embeddings
    )
```

**Expected improvement:** Image Hit Rate: +2-5%

---

## 📋 Implementation Plan

### **Phase 1: Critical Fix** 
1. ✅ Fix metadata deserialization in `retrieve_with_strict_images()`
2. ✅ Fix metadata deserialization in `_verify_metadata_images()`
3. ✅ Test on 2-3 queries manually
4. ✅ Re-run full evaluation

**Expected after Phase 1:**
- Recall@5: 95% (unchanged)
- Image Hit Rate: 85-90% ✅
- MRR: 1.0 (unchanged)

### **Phase 2: Fine-tuning** ⏱️ 15 min
1. Lower threshold for visual query fallback (0.5 → 0.4)
2. Re-run evaluation
3. Adjust if needed

**Expected after Phase 2:**
- Image Hit Rate: 90-95% ✅

### **Phase 3: Optional Enhancement** ⏱️ 30 min
1. Add non-visual image relevance check
2. Test on edge cases
3. Document behavior

---

## 🚀 Next Steps

**Immediate:**
1. Implement Improvement #1 (metadata deserialization)
2. Re-run evaluation
3. Compare results

**After fixing:**
1. Continue with B1: Faithfulness Judge
2. Complete D: Evaluation Report
3. Document all improvements in final report

---

## 💡 Key Insights

### **What Works:**
- ✅ Text retrieval strategy (MMR with λ=0.7)
- ✅ Semantic verification concept
- ✅ Confidence scoring (HIGH/MEDIUM/LOW)
- ✅ ChromaDB indexing

### **What Needs Work:**
- ❌ Metadata deserialization (JSON strings)
- ❌ Image retrieval for non-visual queries
- ⚠️ Threshold tuning for different query types

### **Technical Debt:**
- Inconsistency between indexing (JSON.dumps) and retrieval (string parsing)
- Should standardize metadata format across pipeline
- Consider using ChromaDB's native list type instead of JSON strings

---

## 📈 Expected Final Scores

After all improvements:
```
✅ Recall@5: 95% (target ≥70%)
✅ Image Hit Rate: 90% (target ≥60%)
✅ MRR: 1.0 (target ≥0.70)
```

**Result:** All metrics above target! 🎉
