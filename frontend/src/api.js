import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const uploadPDF = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
};

export const getPageImageUrl = (fileId, pageIndex) => {
    return `${API_BASE_URL}/page/${fileId}/${pageIndex}`;
};

export const processPage = async (fileId, pageIndex, signal = null) => {
    const response = await axios.post(`${API_BASE_URL}/process`, {
        filename: fileId,
        page_index: pageIndex
    }, {
        signal
    });
    return response.data;
};

export const getCroppedImageUrl = (fileId, pageIndex, pdfCoords = null) => {
    let url = `${API_BASE_URL}/crop/${fileId}/${pageIndex}`;
    if (pdfCoords && pdfCoords.length === 4) {
        const [x, y, w, h] = pdfCoords;
        url += `?x=${x}&y=${y}&w=${w}&h=${h}`;
    }
    return url;
};
