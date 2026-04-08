const express = require("express");
const multer = require("multer");
const { spawn } = require("child_process");
const cors = require("cors"); // allow frontend calls

const app = express();
app.use(cors()); // important for cross-origin requests

const upload = multer({ dest: "uploads/" });

app.post("/predict", upload.single("file"), (req, res) => {
    if (!req.file) return res.status(400).send({ error: "No file uploaded" });

    const python = spawn("python", ["model/model.py", req.file.path]);

    let output = "";
    let errorOutput = "";

    python.stdout.on("data", (data) => {
        output += data.toString();
    });

    python.stderr.on("data", (data) => {
        errorOutput += data.toString();
    });

    python.on("close", (code) => {
        if (code !== 0) {
            console.error("Python error:", errorOutput);
            return res.status(500).send({ error: errorOutput || "Python failed" });
        }

        // Send output as JSON
        try {
            const jsonData = JSON.parse(output);
            res.send(jsonData);
        } catch {
            res.send({ result: output.trim() }); // fallback plain text
        }
    });
});

app.listen(5000, () => console.log("Server running on 5000"));