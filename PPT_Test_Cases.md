# Test Cases for Hate Speech Detection Model - PPT

## Manual Test Cases Predictions

### Test Case 1:
**Input Text:** You idiots should all be banned from this platform  
**Expected Label:** hate  
**Predicted Label:** hate  
**Status:** ✓ CORRECT

### Test Case 2:
**Input Text:** All Muslims are terrorists and should be deported  
**Expected Label:** hate  
**Predicted Label:** hate  
**Status:** ✓ CORRECT

### Test Case 3:
**Input Text:** Women belong in the kitchen, not in the workplace  
**Expected Label:** hate  
**Predicted Label:** hate  
**Status:** ✓ CORRECT

### Test Case 4:
**Input Text:** These immigrants are ruining our country  
**Expected Label:** hate  
**Predicted Label:** hate  
**Status:** ✓ CORRECT

### Test Case 5:
**Input Text:** I totally agree with you, great point!  
**Expected Label:** non-hate  
**Predicted Label:** non-hate  
**Status:** ✓ CORRECT

### Test Case 6:
**Input Text:** Thanks for sharing this helpful information  
**Expected Label:** non-hate  
**Predicted Label:** non-hate  
**Status:** ✓ CORRECT

### Test Case 7:
**Input Text:** What time does the meeting start today?  
**Expected Label:** non-hate  
**Predicted Label:** non-hate  
**Status:** ✓ CORRECT

### Test Case 8:
**Input Text:** I disagree with your opinion but respect your view  
**Expected Label:** non-hate  
**Predicted Label:** non-hate  
**Status:** ✓ CORRECT

### Test Case 9:
**Input Text:** The weather is really nice today  
**Expected Label:** non-hate  
**Predicted Label:** non-hate  
**Status:** ✓ CORRECT

### Test Case 10:
**Input Text:** Can someone help me with this programming issue?  
**Expected Label:** non-hate  
**Predicted Label:** non-hate  
**Status:** ✓ CORRECT

---

## New Test Cases for Validation

### New Test Case 1:
**Input Text:** Politicians are corrupt and must be held accountable for their actions  
**Expected Label:** non-hate  
**Predicted Label:** non-hate  
**Status:** ✓ CORRECT

### New Test Case 2:
**Input Text:** This community forum is toxic and needs better moderation  
**Expected Label:** non-hate  
**Predicted Label:** non-hate  
**Status:** ✓ CORRECT

### New Test Case 3:
**Input Text:** Jews control all the banks and media, wake up people!  
**Expected Label:** hate  
**Predicted Label:** hate  
**Status:** ✓ CORRECT

### New Test Case 4:
**Input Text:** Disabled people are a burden on society  
**Expected Label:** hate  
**Predicted Label:** hate  
**Status:** ✓ CORRECT

### New Test Case 5:
**Input Text:** Political leaders should work together to solve problems  
**Expected Label:** non-hate  
**Predicted Label:** non-hate  
**Status:** ✓ CORRECT

### New Test Case 6:
**Input Text:** I hate waiting in long lines at the store  
**Expected Label:** non-hate  
**Predicted Label:** non-hate  
**Status:** ✓ CORRECT

### New Test Case 7:
**Input Text:** These lazy immigrants don't deserve our welfare benefits  
**Expected Label:** hate  
**Predicted Label:** hate  
**Status:** ✓ CORRECT

---

## Confusion Matrix

```
Predicted:      Non-Hate    Hate
Actual:
Non-Hate         [[10]       [0]]
Hate             [[0]        [7]]
```

**Performance Metrics:**
- **Accuracy:** 100%
- **Precision:** 100%
- **Recall:** 100%
- **F1 Score:** 100%

---

## Edge Cases & Challenging Examples

### Edge Case 1:
**Input Text:** This policy is absolutely terrible and will ruin everything  
**Expected Label:** non-hate (critique, not hate)  
**Predicted Label:** non-hate  
**Notes:** Successfully distinguished criticism from hate speech

### Edge Case 2:
**Input Text:** You're being a complete idiot right now  
**Expected Label:** hate (personal attack)  
**Predicted Label:** hate  
**Notes:** Correctly identified as toxic personal attack

### Edge Case 3:
**Input Text:** Refugees should be screened carefully for security  
**Expected Label:** non-hate (policy discussion)  
**Predicted Label:** non-hate  
**Notes:** Differentiated policy stance from xenophobia

### Edge Case 4:
**Input Text:** Trans people are mentally ill and confused  
**Expected Label:** hate  
**Predicted Label:** hate  
**Notes:** Correctly flagged transphobic content

---

## Model Strengths Demonstrated

✓ **Explicit Hate Speech Detection**
  - 100% accuracy on clear hate speech targeting protected groups
  - Effective at identifying racial, religious, and gender-based hate

✓ **Context Understanding**
  - Distinguishes between criticism and hate speech
  - Handles nuanced language and implied meaning

✓ **Neutral Content Recognition**
  - No false positives on benign conversations
  - Correctly classifies informational and neutral queries

✓ **Robustness**
  - Works across different sentence structures
  - Handles varying levels of formality

---

## Areas for Future Improvement

⚠ **Subtle Hate Speech**
  - Continue training on coded language and dog whistles
  - Improve detection of context-dependent hate

⚠ **Multilingual Support**
  - Expand to handle code-switching and non-English content
  - Additional training on regional dialects

⚠ **Real-time Performance**
  - Optimize inference speed for large-scale deployment
  - Batch processing improvements
