//Server.js

const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const cors = require('cors');

const app = express();
const PORT = 5003;

app.use(cors());
app.use(bodyParser.json());

// Route for processing text input from Page2
app.post('/api/process-text', async (req, res) => {
    const { text } = req.body;
    console.log("Received text in Node.js server:", text);

    if (!text || text.trim().length === 0) {
        return res.status(400).json({ error: 'Text input is empty.' });
    }

    try {
        console.log("Forwarding text to Python server...");
        const response = await axios.post('http://localhost:5004/process-text', { text });
        res.json(response.data);
    } catch (error) {
        console.error('Error calling Python server:', error.response?.data || error.message);
        res.status(500).json({ error: error.response?.data?.error || error.message });
    }
});


// Route for processing link input from Page1
app.post('/api/fetch-data', async (req, res) => {
    const { url } = req.body;
    console.log("Received URL in Node.js server:", url);

    if (!url) {
        return res.status(400).json({ error: 'URL is required' });
    }

    try {
        console.log("Forwarding URL to Python server...");
        const response = await axios.post('http://localhost:5004/fetch-data', { url });
        res.json(response.data);
    } catch (error) {
        console.error('Error calling Python server:', error.message);
        res.status(500).json({ error: error.message });
    }
});

// Route for processing thread link input
app.post('/api/fetch-thread-data', async (req, res) => {
    const { url } = req.body;
    console.log("Received Thread URL in Node.js server:", url);

    if (!url) {
        return res.status(400).json({ error: 'Thread URL is required' });
    }

    try {
        console.log("Forwarding thread URL to Python server...");
        const response = await axios.post('http://localhost:5004/fetch-thread-data', { url });
        res.json(response.data);
    } catch (error) {
        console.error('Error calling Python server for thread data:', error.message);
        res.status(500).json({ error: error.message });
    }
});

//Route for fetching and processing thread data
app.post('/api/fetch-threads', async (req, res) => {
    const { url } = req.body;
    console.log("Received Thread URL in Node.js server:", url);

    if (!url) {
        return res.status(400).json({ error: 'URL is required' });
    }

    try {
        console.log("Forwarding thread URL to Python server...");
        const response = await axios.post('http://localhost:5004/fetch-threads', { url });
        res.json(response.data);
    } catch (error) {
        console.error('Error calling Python server for thread data:', error.message);
        res.status(500).json({ error: error.message });
    }
});


app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});



