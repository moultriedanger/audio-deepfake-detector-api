<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h1 align="center">Fake Or Real? – An Audio Deepfake Detection Api</h1>
  <p align="center">
    Backend for a full-stack audio deepfake detection applicaiton that utilizes a fine-tuned hugging face transformer to detect AI-generated speech.
    <br />
    Built to help identify deepfake audio in real time, with an intuitive upload interface and clear prediction results.
    <br />
    <br />
    <a href="https://audio-deepfake-detector.vercel.app/"><strong>Live Demo »</strong></a>
    <br />
    <br />
    <a href="https://github.com/moultriedanger/audio-deepfake-detector-frontend">Frontend Repo</a>
    &middot;
    <a href="https://github.com/moultriedanger/audio-deepfake-detector-api">Backend Repo</a>
  </p>
</div>

---

![Front End Screenshot](images/record_result.png)

## 📌 Technologies Used

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="40" alt="Python" title="Python" />
  </a>
  <a href="https://flask.palletsprojects.com/">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/flask/flask-original.svg" height="40" alt="Flask" title="Flask" />
  </a>
  <a href="https://pytorch.org/">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" height="40" alt="PyTorch" title="PyTorch" />
  </a>
  <a href="https://numpy.org/">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" height="40" alt="NumPy" title="NumPy" />
  </a>
  <a href="https://huggingface.co/">
    <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/huggingface.svg" height="40" alt="Hugging Face" title="Hugging Face" />
  </a>
   <a href="https://gunicorn.org/">
    <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/gunicorn.svg" height="40" alt="Gunicorn" title="Gunicorn" />
  </a>
  <a href="https://www.heroku.com/">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/heroku/heroku-original.svg" height="40" alt="Heroku" title="Heroku" />
  </a>
</p>

## 🚀 Usage

### POST `/predict`
Run deepfake detection on an uploaded audio file.

- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Body:**
  - `file`: audio file to analyze (e.g., `.wav`, `.mp3`). WAV recommended.
- **Response (200):**
  ```json
  {
    "prediction": "fake",
    "scores": {
      "fake": 0.9983661770820618,
      "real": 0.0016337810084223747
    }
  }

