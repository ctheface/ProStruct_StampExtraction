# ProStruct: Stamp Extractor

A web application that automatically detects and extracts engineer approval stamps from PDF structural drawings. Uses computer vision (Hough Circle Detection) and OCR to identify circular stamps, extract engineer names and license numbers, and display results with visual overlays.

![ProStruct UI — PDF viewer with search region overlay](images/Screenshot%202026-07-01%20132842.png)

![Extraction results — detected stamps with engineer name and license](images/Screenshot%202026-07-01%20132933.png)

## 🎯 Features

- **PDF Upload & Page Selection**: Upload PDF drawings and navigate through pages
- **Automatic Stamp Detection**: Multi-scale Hough Circle Detection finds circular engineer stamps
- **Smart Region Detection**: Focuses on the right 40% × top 70% where stamps typically appear
- **Center-Crop OCR**: Extracts text from the center of stamps (avoiding curved perimeter text)
- **Name & License Extraction**: Combines multi-line names and extracts license patterns
- **Visual Overlays**: Displays bounding boxes on detected stamps with search region preview
- **Structured JSON Output**: Returns clean, structured data with coordinates, names, and license numbers

## 🏗️ Project Structure

```
ProStruct/
├── backend/                 # FastAPI backend server
│   ├── main.py             # Main API server and detection logic
│   ├── requirements.txt    # Python dependencies
│   └── temp_uploads/       # Temporary PDF storage
├── frontend/               # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx        # Main React component
│   │   ├── api.js         # API client functions
│   │   └── index.css      # Styles
│   ├── package.json       # Node.js dependencies
│   └── vite.config.js     # Vite configuration
└── README.md              # This file
```

## 🛠️ Technologies Used

| Component | Technology | Purpose |
|-----------|------------|---------|
| Backend | FastAPI + Python | REST API server |
| PDF Processing | PyMuPDF (fitz) | PDF to image conversion at 150 DPI |
| Computer Vision | OpenCV | Hough Circle Detection, image preprocessing |
| OCR | OCR.space API | Text extraction from stamp centers |
| Frontend | React 19 + Vite | User interface |

## 🔍 How Detection Works

```
PDF → Image (150 DPI) → Search Region (right 40%, top 70%)
                                     ↓
                        Multi-scale Hough Circle Detection
                        (radius ranges: 30-60, 50-100, 80-150, 120-200px)
                                     ↓
                        Circularity Verification (edge-mask overlap)
                                     ↓
                        Center Crop (inner 60% of stamp for OCR)
                                     ↓
                        Name Assembly (combine lines: "MARY E." + "DANIELSON")
                                     ↓
                        License Pattern Matching ("No. 55926")
```

### Key Insight: Center-Crop OCR
The curved text around the stamp perimeter produces garbage OCR. By cropping just the **center 60%** where text is straight, we get clean extraction:
- Center contains: FIRST NAME / LAST NAME / CIVIL or ENVIRONMENTAL / No. XXXXX

## 📋 Prerequisites

1. **Python 3.8+** ([Download](https://www.python.org/downloads/))
2. **Node.js 16+** and npm ([Download](https://nodejs.org/))
3. **OCR.space API Key** (optional): Get free key from [OCR.space](https://ocr.space/ocrapi/freekey)

## 🚀 Quick Start

### Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # macOS/Linux
pip install -r requirements.txt

# Optional: Create .env with your API key
echo OCR_SPACE_API_KEY=your_key_here > .env

python main.py                # Starts on http://localhost:8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev                   # Starts on http://localhost:5173
```

## 📖 Usage

1. **Upload PDF**: Select a PDF structural drawing
2. **Select Page**: Navigate to page with engineer stamps
3. **Detect Stamps**: Click "Detect Stamp" button
4. **View Results**: See bounding boxes, cropped previews, and extracted data

## 📄 Output Format

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

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload PDF, returns file_id and page count |
| `/page/{file_id}/{page_index}` | GET | Get page image (PNG) |
| `/process` | POST | Detect stamps and extract info |
| `/crop/{file_id}/{page_index}` | GET | Get cropped stamp region |

## 🐛 Troubleshooting

- **OCR errors**: Check API key in `.env`, ensure internet connection
- **Port in use**: Change port in `main.py` or `vite.config.js`
- **No stamps detected**: Stamps must be circular; try different pages
