
<div align="center">

![Bharat AI Pro Logo](images/logo.png)



# Bharat AI Pro




### India’s Indigenous AI Assistant for Healthcare, Education, Productivity, and Hyperlocal Services

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-3910/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/bharat-ai-pro?style=social)](https://github.com/yourusername/bharat-ai-pro/stargazers)

**Bharat AI Pro** is a cutting-edge, open-source artificial intelligence framework designed specifically for the Indian context. Leveraging the power of local Large Language Models (LLMs) and lightweight APIs, it provides secure, offline-capable, and hyperlocal solutions for healthcare diagnostics, educational tutoring, and productivity enhancement.


[🚀 Get Started](#installation) • [📖 Documentation](#usage) • [🤝 Contributing](#contributing) • [🐛 Report Bug](https://github.com/yourusername/bharat-ai-pro/issues)

</div>

---

## 📸 Project Overview

### Web Interface Demo
A clean, responsive user interface designed for accessibility across devices.

![Web Demo](images/web_demo.png)

### AI Chat Interaction
Experience seamless multilingual conversations with the context-aware AI assistant.

![Chat Demo](images/chat_demo.png)

### Dashboard & Analytics
Monitor usage, system health, and AI performance metrics in real-time.

![Dashboard Overview](images/dashboard.png)

---

## ✨ Key Features

- **🏥 Healthcare Assistant**: Preliminary diagnostic suggestions and health tips based on symptoms (for educational purposes).
- **🎓 Education Tutor**: Personalized learning support for students in regional languages.
- **💬 Hyperlocal Services**: Integration with local data to provide relevant information on services, weather, and agriculture.
- **🔒 Privacy First**: Capable of running entirely offline using local LLMs (Ollama/Llama.cpp) ensuring data sovereignty.
- **🌐 Multilingual Support**: Optimized for Indian languages including Hindi, Tamil, Bengali, and Marathi.
- **⚡ Fast & Lightweight**: Built on Flask and Vanilla JS to ensure performance on low-end hardware.

---

## 🛠 Tech Stack

- **Backend**: Python 3.9+, Flask
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **AI/ML Engine**: Ollama (Llama 3 / Mistral), Transformers
- **Database**: SQLite (for local history), JSON
- **APIs**: OpenAI API (Optional fallback), LangChain

---

## 📥 Installation

To run **Bharat AI Pro** locally, follow these steps:

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Ollama (for local LLM support) [Download here](https://ollama.com/)



### Clone the Repository
```bash
git clone https://github.com/yourusername/bharat-ai-pro.git
cd bharat-ai-pro
```

### Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Setup AI Model (Optional but Recommended)
If you want to use the offline LLM:
```bash
ollama pull llama3
```

---

## 🚀 Usage



1. **Start the Flask Server**:
   ```bash
   python app.py
   ```
   The application will start running at `http://127.0.0.1:5000`.

2. **Open in Browser**:
   Navigate to `http://127.0.0.1:5000` in your web browser.

3. **Interact with the AI**:
   - Select a mode (Healthcare, Education, General).
   - Type your query in English or a supported regional language.
   - View the AI response and utilize the dashboard for insights.

---

## 🤝 Contributing

We welcome contributions from developers across India and the globe! Here's how you can help:

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

Please read our `CODE_OF_CONDUCT.md` for details on our code of conduct.

---


## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---



<div align="center">


**Made with ❤️ by Rohit Pawar **

[⬆ Back to Top](#bharat-ai-pro)

</div>

