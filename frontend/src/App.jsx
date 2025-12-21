import React, { useState, useEffect, useRef } from 'react';
import { uploadPDF, getPageImageUrl, processPage, getCroppedImageUrl } from './api';

function App() {
  const [fileId, setFileId] = useState(null);
  const [pageCount, setPageCount] = useState(0);
  const [currentPage, setCurrentPage] = useState(0);
  const [loading, setLoading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const imgRef = useRef(null);
  const abortControllerRef = useRef(null);
  const [imgDimensions, setImgDimensions] = useState({ width: 0, height: 0, naturalWidth: 0, naturalHeight: 0 });

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    try {
      const data = await uploadPDF(file);
      setFileId(data.file_id);
      setPageCount(data.page_count);
      setCurrentPage(0);
      setResults(null);
    } catch (err) {
      console.error(err);
      setError("Failed to upload PDF");
    } finally {
      setLoading(false);
    }
  };

  const handlePageChange = (newPage) => {
    setCurrentPage(newPage);
    setResults(null); // Reset results on page change
  };

  const runExtraction = async () => {
    // Cancel previous request if exists
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    setProcessing(true);
    setError(null);

    try {
      const data = await processPage(fileId, currentPage, abortControllerRef.current.signal);
      setResults(data);
    } catch (err) {
      if (err.name === 'CanceledError' || err.code === 'ERR_CANCELED') {
        console.log('Detection cancelled');
      } else {
        console.error(err);
        setError("Failed to process page. Please check if the backend server is running and try again.");
      }
    } finally {
      if (abortControllerRef.current && !abortControllerRef.current.signal.aborted) {
        setProcessing(false);
      }
      // If manually aborted, processing state handled by cancel button
    }
  };

  const cancelExtraction = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setProcessing(false);
    }
  };

  // Update image dimensions for scaling the bounding box
  const onImgLoad = (e) => {
    setImgDimensions({
      width: e.target.offsetWidth,
      height: e.target.offsetHeight,
      naturalWidth: e.target.naturalWidth,
      naturalHeight: e.target.naturalHeight,
    });
  };

  // Recalculate on window resize
  useEffect(() => {
    const handleResize = () => {
      if (imgRef.current) {
        setImgDimensions({
          width: imgRef.current.offsetWidth,
          height: imgRef.current.offsetHeight,
          naturalWidth: imgRef.current.naturalWidth,
          naturalHeight: imgRef.current.naturalHeight,
        });
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [results]);


  // Colors for multiple stamp bounding boxes
  const boxColors = [
    { border: '#ef4444', bg: 'rgba(239, 68, 68, 0.15)' }, // red
    { border: '#3b82f6', bg: 'rgba(59, 130, 246, 0.15)' }, // blue
    { border: '#22c55e', bg: 'rgba(34, 197, 94, 0.15)' },  // green
    { border: '#f59e0b', bg: 'rgba(245, 158, 11, 0.15)' }, // amber
    { border: '#8b5cf6', bg: 'rgba(139, 92, 246, 0.15)' }, // purple
  ];

  // Calculate Box Style for a specific stamp
  const getBoxStyle = (stamp, index) => {
    if (!stamp || !stamp.bounding_box || imgDimensions.naturalWidth === 0) return { display: 'none' };

    const [x, y, w, h] = stamp.bounding_box;
    const scaleX = imgDimensions.width / imgDimensions.naturalWidth;
    const scaleY = imgDimensions.height / imgDimensions.naturalHeight;
    const color = boxColors[index % boxColors.length];

    return {
      left: `${x * scaleX}px`,
      top: `${y * scaleY}px`,
      width: `${w * scaleX}px`,
      height: `${h * scaleY}px`,
      position: 'absolute',
      border: `3px solid ${color.border}`,
      backgroundColor: color.bg,
      zIndex: 10,
      pointerEvents: 'none'
    };
  };

  // Calculate Search Region Overlay Style
  const getSearchRegionStyle = () => {
    if (!results || imgDimensions.naturalWidth === 0) return { display: 'none' };

    // Get search region from results (could be single object or array)
    let searchRegion = null;
    if (!Array.isArray(results) && results.search_region) {
      searchRegion = results.search_region;
    } else if (Array.isArray(results) && results.length > 0 && results[0].search_region) {
      searchRegion = results[0].search_region;
    }

    if (!searchRegion) return { display: 'none' };

    const [x, y, w, h] = searchRegion;
    const scaleX = imgDimensions.width / imgDimensions.naturalWidth;
    const scaleY = imgDimensions.height / imgDimensions.naturalHeight;

    return {
      left: `${x * scaleX}px`,
      top: `${y * scaleY}px`,
      width: `${w * scaleX}px`,
      height: `${h * scaleY}px`,
      position: 'absolute',
      border: `2px dashed #6366f1`, // Indigo dashed border
      backgroundColor: 'rgba(99, 102, 241, 0.1)', // Light indigo background
      zIndex: 5, // Below stamp boxes but visible
      pointerEvents: 'none'
    };
  };


  return (
    <div className="min-h-screen bg-gray-100 p-8 text-slate-800">
      <div className="max-w-6xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">ProStruct: Stamp Extractor</h1>
          <p className="text-gray-600">Automated Approval Stamp Detection & OCR</p>
        </header>

        {/* UPLOAD SECTION */}
        {!fileId && (
          <div className="bg-white p-12 rounded-xl shadow-lg text-center border-2 border-dashed border-gray-300">
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileUpload}
              className="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-full file:border-0
                file:text-sm file:font-semibold
                file:bg-blue-50 file:text-blue-700
                hover:file:bg-blue-100
                cursor-pointer
              "
            />
            {loading && <p className="mt-4 text-blue-600">Uploading...</p>}
            {error && <p className="mt-4 text-red-600">{error}</p>}
          </div>
        )}

        {/* WORKSPACE */}
        {fileId && (
          <div className="space-y-6">

            {/* VIEWER SECTION */}
            <div className="space-y-4">
              {/* Toolbar */}
              <div className="flex items-center justify-between bg-white p-4 rounded-lg shadow">
                <div className="flex items-center space-x-4">
                  <button
                    disabled={currentPage <= 0}
                    onClick={() => handlePageChange(currentPage - 1)}
                    className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50 hover:bg-gray-300 transition"
                  >
                    ← Prev
                  </button>

                  {/* Page Selector Dropdown */}
                  <div className="flex items-center space-x-2">
                    <label className="text-sm font-medium text-gray-700">Page:</label>
                    <select
                      value={currentPage}
                      onChange={(e) => handlePageChange(parseInt(e.target.value))}
                      className="border-gray-300 rounded shadow-sm focus:border-blue-500 focus:ring-blue-500"
                    >
                      {Array.from({ length: pageCount }, (_, i) => (
                        <option key={i} value={i}>
                          {i + 1}
                        </option>
                      ))}
                    </select>
                    <span className="text-sm text-gray-500">of {pageCount}</span>
                  </div>

                  <button
                    disabled={currentPage >= pageCount - 1}
                    onClick={() => handlePageChange(currentPage + 1)}
                    className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50 hover:bg-gray-300 transition"
                  >
                    Next →
                  </button>
                </div>

                <div className="flex items-center space-x-2">
                  {processing ? (
                    <button
                      onClick={cancelExtraction}
                      className="px-4 py-2 bg-red-600 text-white rounded font-medium hover:bg-red-700 transition flex items-center gap-2 shadow-sm"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
                      </svg>
                      Stop
                    </button>
                  ) : (
                    <button
                      onClick={runExtraction}
                      disabled={processing}
                      className="px-4 py-2 bg-blue-600 text-white rounded font-medium hover:bg-blue-700 transition flex items-center gap-2 shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                      </svg>
                      Detect Stamp
                    </button>
                  )}
                </div>
              </div>

              {/* Image Canvas */}
              <div className="relative bg-white p-2 rounded-lg shadow overflow-hidden flex justify-center border border-gray-200" style={{ minHeight: '500px' }}>
                <div className="relative inline-block">
                  <img
                    ref={imgRef}
                    src={getPageImageUrl(fileId, currentPage)}
                    alt={`Page ${currentPage + 1}`}
                    onLoad={onImgLoad}
                    className="max-w-full h-auto shadow-sm"
                    style={{ maxHeight: '70vh' }}
                  />
                  {/* Search Region Overlay - shows where OCR search is performed */}
                  {results && (
                    <div 
                      style={getSearchRegionStyle()} 
                      title="OCR Search Region (Right 40%, Top 70%)"
                    />
                  )}
                  {/* Bounding Box Overlay - handles both single and array response */}
                  {results && !Array.isArray(results) && results.bounding_box && (
                    <div style={getBoxStyle(results, 0)} title={`Detected: ${results.engineer_name}`} />
                  )}
                  {results && Array.isArray(results) && results.map((stamp, idx) => (
                    <div key={idx} style={getBoxStyle(stamp, idx)} title={`Stamp ${idx + 1}: ${stamp.engineer_name}`} />
                  ))}
                </div>
              </div>
            </div>

            {/* EXTRACTION RESULTS SECTION - Below the PDF */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-xl font-bold mb-4 border-b pb-2">Extraction Results</h2>

              {!results && !processing && (
                <p className="text-gray-400 italic">Click "Detect Stamp" to analyze this page.</p>
              )}

              {processing && (
                <div className="flex items-center space-x-2 text-blue-600">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                  </svg>
                  <span>Analyzing layout and OCR...</span>
                </div>
              )}

              {/* Single stamp result */}
              {results && !Array.isArray(results) && results.bounding_box && (
                <div className="space-y-6">
                  {/* Stamp card */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <div className="border-2 border-red-400 rounded-lg p-4 space-y-3">
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-4 h-4 rounded bg-red-500"></div>
                        <h3 className="font-bold">Detected Stamp</h3>
                      </div>
                      <div className="space-y-2 text-sm">
                        <div>
                          <label className="text-xs text-gray-500 uppercase">Engineer</label>
                          <div className="font-medium text-lg">{results.engineer_name || "Unknown"}</div>
                        </div>
                        <div>
                          <label className="text-xs text-gray-500 uppercase">License #</label>
                          <div className="font-medium text-lg">{results.license_number || "Unknown"}</div>
                        </div>
                      </div>
                      <div>
                        <label className="text-xs text-gray-500 uppercase block mb-1">Cropped Preview</label>
                        <img src={getCroppedImageUrl(fileId, currentPage, results.bounding_box)} alt="Stamp" className="max-w-full h-auto rounded border" />
                      </div>
                    </div>
                  </div>
                  {/* JSON Output */}
                  <div>
                    <label className="block text-xs font-semibold text-gray-500 uppercase mb-2">JSON Output</label>
                    <pre className="text-xs bg-gray-900 text-green-400 p-3 rounded-lg overflow-x-auto font-mono">
                      {JSON.stringify({ page: results.page, symbol_type: "approval_stamp", bounding_box: results.bounding_box, engineer_name: results.engineer_name, license_number: results.license_number, units: "pixels" }, null, 2)}
                    </pre>
                  </div>
                </div>
              )}

              {/* Multiple stamps result */}
              {results && Array.isArray(results) && (
                <div className="space-y-6">
                  <div className="p-2 bg-blue-50 text-blue-800 text-sm rounded-lg border border-blue-200">
                    <strong>📋 Found {results.length} stamps</strong>
                  </div>
                  {/* Stamps grid - side by side */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {results.map((stamp, idx) => {
                      const color = boxColors[idx % boxColors.length];
                      return (
                        <div key={idx} className="border-2 rounded-lg p-4 space-y-3" style={{ borderColor: color.border }}>
                          <div className="flex items-center gap-2">
                            <div className="w-4 h-4 rounded" style={{ backgroundColor: color.border }}></div>
                            <h3 className="font-bold">Stamp {idx + 1}</h3>
                          </div>
                          <div className="space-y-2 text-sm">
                            <div>
                              <label className="text-xs text-gray-500 uppercase">Engineer</label>
                              <div className="font-medium">{stamp.engineer_name || "Unknown"}</div>
                            </div>
                            <div>
                              <label className="text-xs text-gray-500 uppercase">License #</label>
                              <div className="font-medium">{stamp.license_number || "Unknown"}</div>
                            </div>
                          </div>
                          <div>
                            <label className="text-xs text-gray-500 uppercase block mb-1">Cropped Preview</label>
                            <img src={getCroppedImageUrl(fileId, currentPage, stamp.bounding_box)} alt={`Stamp ${idx + 1}`} className="max-w-full h-auto rounded border" />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  {/* JSON Output */}
                  <div>
                    <label className="block text-xs font-semibold text-gray-500 uppercase mb-2">JSON Output</label>
                    <pre className="text-xs bg-gray-900 text-green-400 p-3 rounded-lg overflow-x-auto font-mono">
                      {JSON.stringify(results, null, 2)}
                    </pre>
                  </div>
                </div>
              )}

              {error && (
                <div className="mt-4 p-3 bg-red-50 text-red-700 rounded border border-red-200 text-sm">
                  {error}
                </div>
              )}
            </div>

            <button
              onClick={() => setFileId(null)}
              className="text-gray-500 hover:text-gray-700 text-sm w-full text-center"
            >
              Upload a different PDF
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;