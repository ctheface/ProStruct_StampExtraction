# ProStruct Stamp Extractor - How It Works

This document explains the end-to-end pipeline for detecting engineer stamps and extracting information from structural drawings.

---

## System Overview

```mermaid
flowchart LR
    A[PDF Upload] --> B[Page Rendering]
    B --> C[Search Region]
    C --> D[Circle Detection]
    D --> E[OCR Processing]
    E --> F[Name/License Extraction]
    F --> G[JSON Output]
```

---

## Step 1: PDF Processing

When a PDF is uploaded:

1. **PyMuPDF** (`fitz`) converts the selected page to an image at **150 DPI**
2. The image is loaded into OpenCV as a NumPy array
3. Image dimensions are captured for coordinate calculations

```python
# Render PDF page to image
pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
```

---

## Step 2: Search Region Definition

Stamps are typically in the **title block** (bottom-right). We define a search region:

| Parameter | Value | Description |
|-----------|-------|-------------|
| X Start | 60% of width | Right 40% of page |
| Y Start | 0 | Top of page |
| Width | 40% of page | Covers title block area |
| Height | 70% of page | Top portion where stamps appear |

![Search Region](file:///C:/Users/Manas%20Budhiraja/.gemini/antigravity/brain/98a09303-d1dc-44c3-a794-7b897098755d/uploaded_image_0_1766820968284.png)

---

## Step 3: Circle Detection (Hough Transform)

Engineer stamps are **circular seals**. We use OpenCV's Hough Circle Detection:

```python
circles = cv2.HoughCircles(
    gray_image,
    cv2.HOUGH_GRADIENT,
    dp=1.5,              # Inverse resolution ratio
    minDist=min_radius,  # Min distance between circles
    param1=80,           # Canny edge threshold
    param2=40,           # Accumulator threshold
    minRadius=30,        # Smallest stamp radius
    maxRadius=200        # Largest stamp radius
)
```

### Multi-Scale Detection

We search 4 radius ranges to find stamps of different sizes:
- **30-60px**: Small stamps
- **50-100px**: Medium stamps
- **80-150px**: Large stamps
- **120-200px**: Very large stamps

### Validation Filters

Each detected circle is validated:

1. **Edge Density**: Must have ≥3% edges (rejects empty areas)
2. **Circularity Score**: Edges must align with circular pattern
3. **Duplicate Removal**: Distance-based filtering to avoid overlaps

---

## Step 4: OCR Strategy - Center Crop

**The Key Insight**: Circular stamps have curved text around the perimeter that produces garbage OCR. The **center** has straight, readable text.

```
        ┌─────────────────────────────┐
        │  COMMONWEALTH OF MASS...    │ ← Curved (SKIP!)
        │    ┌─────────────────┐      │
        │    │   MARY E.       │      │
        │    │   DANIELSON     │      │ ← Center (OCR THIS!)
        │    │   ENVIRONMENTAL │      │
        │    │   No. 55926     │      │
        │    └─────────────────┘      │
        │  ...PROFESSIONAL ENGINEER   │ ← Curved (SKIP!)
        └─────────────────────────────┘
```

We crop the **inner 60%** of the stamp (20% margin on each side):

```python
center_margin = int(stamp_size * 0.2)  # 20% margin
center_region = img[y+margin : y+h-margin, x+margin : x+w-margin]
```

---

## Step 5: Name Extraction Logic

The OCR returns text line-by-line. We process it:

### Line Cleaning

1. Remove lines with excluded keywords: `CIVIL`, `ENVIRONMENTAL`, `REGISTERED`, etc.
2. Remove lines with numbers (license line)
3. Keep only lines with ≥80% alphabetic characters

### Name Assembly

Names often appear on **separate lines**:

```
OCR Output:          Clean Lines:       Combined:
MARY E.         →    ["MARY E.",    →   "MARY E. DANIELSON"
DANIELSON            "DANIELSON"]
ENVIRONMENTAL
No. 55926
```

```python
# Combine consecutive lines
first_part = "MARY E."      # 1-2 words
second_part = "DANIELSON"   # 1 word
combined = f"{first_part} {second_part}"  # "MARY E. DANIELSON"
```

---

## Step 6: License Number Extraction

We look for patterns like:

```python
patterns = [
    r'No\.?\s*(\d{4,6})',     # "No. 55926"
    r'#\s*(\d{4,6})',         # "# 55926"
    r'License.*?(\d{4,6})',   # "License 55926"
]
```

---

## Final Output

The system returns JSON with extracted data:

```json
[
  {
    "page": 1,
    "symbol_type": "approval_stamp",
    "bounding_box": [4679, 1068, 108, 108],
    "engineer_name": "MARY E. DANIELSON",
    "license_number": "55926",
    "units": "pixels"
  },
  {
    "page": 1,
    "symbol_type": "approval_stamp", 
    "bounding_box": [4735, 534, 118, 118],
    "engineer_name": "THOMAS MAHANNA",
    "license_number": "39479",
    "units": "pixels"
  }
]
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Backend | FastAPI + Python | REST API server |
| PDF Processing | PyMuPDF (fitz) | PDF to image conversion |
| Computer Vision | OpenCV | Circle detection, image processing |
| OCR | OCR.space API | Text extraction from images |
| Frontend | React | User interface |
