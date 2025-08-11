# AI Notebooks Portfolio

This repository showcases a curated selection of AI projects and experiments developed between March 2023 and August 2025.  
It's organized into three sections:

1. **Showcase** – polished, production-ready notebooks demonstrating key skills  
2. **Experiments** – exploratory code and prototype work  
3. **Archive** – retired or hidden notebooks

---

## Featured Projects

### 🏆 HR Assistant with RAG
**2023-03_HR_RAG_Chatbot_polished.ipynb**  
A production-ready HR chatbot built for The Washington Post's internal HR portal. Uses Retrieval-Augmented Generation with OpenAI's GPT-4 to answer employee questions about benefits, policies, and procedures. Handles 2300+ HR articles with vector embeddings for accurate information retrieval.

**Key Features:**
- Semantic search using embeddings
- Confidence scoring to prevent hallucinations
- Interactive Gradio interface
- Source attribution for transparency
- Handles 2300+ HR articles with efficient vector embeddings

**Technical Highlights:**
- Uses OpenAI's text-embedding-ada-002 for document embeddings
- Employs cosine similarity for semantic search
- Implements confidence thresholding to ensure accurate responses
- Includes a web-based interface using Gradio

### 🌉 Base Model Chat Bridge
**[2025-04_BaseModel-Chat-Bridge.ipynb](https://github.com/kutyadog/ai_notebooks/blob/main/showcase/2025-04_BaseModel-Chat-Bridge.ipynb)**  
Proof-of-concept for bridging base models with chat interactions using an instruct model as an "interpreter" to reformat user questions for specialized base models. Demonstrates how to enable conversational use of non-chat-optimized models like Cisco's Foundation-Sec-8B.

**Key Features:**
- Base-to-instruct model architecture
- Multi-model conversation flow
- Interactive Gradio interface
- Confidence scoring system
- Performance metrics analysis
diff>

### 🤖 Multi-Agent RAG System
**2024-01_AgentChatGroupChatRAG.ipynb**  
Advanced multi-agent system using Microsoft AutoGen framework with Retrieval-Augmented Generation capabilities. Demonstrates complex task delegation and coordination between specialized AI agents.

**Key Features:**
- Multiple specialized agents with distinct roles
- RAG integration for knowledge retrieval
- Dynamic task assignment and coordination
- Wikipedia and custom tool integration

### 🔍 AI-Powered SEO Analysis
**2023-12_AI_SEO_POC.ipynb**  
Comprehensive SEO analysis tool that evaluates content optimization, suggests improvements, and provides actionable insights for better search engine rankings.

**Key Features:**
- Automated SEO scoring (headline, meta tags, content)
- AI-powered content optimization suggestions
- Google Trends integration for keyword research
- Interactive dashboard with detailed recommendations

### 📚 RAG Basics Tutorial
**2024-08_AiRAGBasics.ipynb**  
Step-by-step implementation guide for building a Retrieval-Augmented Generation system from scratch. Perfect for understanding the fundamentals of RAG architecture.

**Key Features:**
- Clear, commented implementation
- Embedding generation and storage
- Semantic search functionality
- Complete end-to-end pipeline

---

## All Showcase Projects

### Conversational AI & Chatbots
- **2023-03_HR_RAG_Chatbot_polished.ipynb** - HR-focused chatbot with RAG
- **2024-01_AgentChatGroupChatRAG.ipynb** - Multi-agent group chat system
- **2025-04_BaseModel-Chat-Bridge.ipynb** - Base model to chat interface bridge
diff>
- **2023-10_ChatbotSimpleRAG.ipynb** - Lightweight RAG chatbot prototype
- **2023-07_LangChainChatGPT.ipynb** - Early LangChain integration experiments

### RAG & Document Processing
- **2024-08_AiRAGBasics.ipynb** - Step-by-step RAG implementation tutorial
- **2023-03_EmbeddingsDocxRAG.ipynb** - Document embedding and retrieval

### SEO & Content Analysis
- **2023-12_AI_SEO_POC.ipynb** - AI-driven SEO analysis tool
- **2023-12_AI_SEO_Tests.ipynb** - SEO optimization experiments

### Exploratory Projects
- **2023-08_CampingRawExploration.ipynb** - Generative AI exploration

---

## Experiments

A collection of exploratory notebooks testing various AI/ML techniques:

### Fine-tuning Projects
- **Fine_Tune_DeepSeek_R1_custom_data_test.ipynb** - Custom data fine-tuning
- **Finetune_redpajama_tests.ipynb** - RedPajama model experimentation
- **Distill_Model_FT.ipynb** - Model distillation techniques
- **Unsloth_finetuning_CJ.ipynb** - Efficient fine-tuning with Unsloth
- **Phi_3_Mini_4K_Instruct_Unsloth_2x_faster_finetuning_TEST.ipynb** - Fast Phi-3 fine-tuning
- **Falcon_7b_finetune_midjourney_falcon_test.ipynb** - Falcon model experimentation
- **rlhf_tune_llm_CJ.ipynb** - RLHF tuning implementation
- **RLHF_with_Custom_Datasets_CJ.ipynb** - Custom RLHF datasets

### Model Testing & Evaluation
- **OpenAI_Chatbot_test.ipynb** - Chatbot model testing
- **llama_langchain_chatbot_custom_data_CJ.ipynb** - LLaMA integration
- **LangChain_Model_Laboratory.ipynb** - LangChain framework exploration

### Specialized Applications
- **Kaggle_skin_cancer.ipynb** - Medical image classification
- **Neural_Networks_pyTorch_CJ.ipynb** - Deep learning fundamentals
- **OpenAI_Text_To_Speech.ipynb** - Audio generation
- **Whisper_voices_cj.ipynb** - Speech recognition

### API Integration & Tools
- **FastAPI_localtunnel_CJ.ipynb** - API deployment
- **gmail_api_test_cj.ipynb** - Gmail API integration
- **Semantic_search_openai_CJ.ipynb** - Semantic search implementation
- **CJ_Snippets_Drive.ipynb** - Google Drive integration
- **linkboxes_test_cj.ipynb** - Link processing utilities

### Creative & Experimental
- **Clone_your_own_voice_tortoise_tts_CJ.ipynb** - Voice cloning with TTS
- **Sentiment_analysis_tiny.ipynb** - Sentiment analysis implementation
- **SEO_Analysis_ORIG_version.ipynb** - Original SEO analysis concept
- **TensorBoard_fashion_test.ipynb** - Model visualization
- **RouteLLM_test.ipynb** - Routing model testing

*(See the `experiments/` folder for the complete list.)*

---

## Archive

Retired or duplicate notebooks:
- **Docker_colab_Python_for_SEO.ipynb**

---

## Timeline

- **August 2025**: Current portfolio organization
- **April 2025**: Base Model Chat Bridge, Multi-Agent RAG System
- **December 2023**: SEO Analysis Tool development
- **March 2023**: Initial HR Chatbot implementation

---

## Setup & Running

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- API keys for various services (OpenAI, Groq, etc.)

### Installation
1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/your-repo.git
   ```

2. Install dependencies:  
   ```bash
   pip install jupyterlab openai langchain numpy pandas scikit-learn matplotlib gradio transformers torch pinecone-client tiktoken datasets
   ```

3. Set your API keys as environment variables:  
   ```bash
   # Required for most projects
   export OPENAI_API_KEY="your_openai_key"
   export OPENAI_ORG="your_org_id"
   
   # For Cybersecurity AI Assistant
   export GROQ_API_KEY="your_groq_key"
   
   # For SEO Analysis Tool
   export GOOGLE_API_KEY="your_google_key"
   export WAPO_PRISM_KEY="your_wapo_key"
   ```

4. Launch Jupyter lab or notebook:  
   ```bash
   jupyter lab
   ```

5. Navigate to the `showcase/` or `experiments/` folder and open any notebook.

### Project-Specific Setup
- **HR Chatbot**: Requires formatted_articles.csv (HR content)
- **Base Model Chat Bridge**: Requires internet connection for model downloads
- **SEO Analysis**: Requires VPN access to internal Washington Post APIs
- **RAG Basics**: No additional setup required - self-contained example

---

## Skills Demonstrated

- **Natural Language Processing**: RAG, embeddings, chatbot development
- **Machine Learning**: Fine-tuning, model evaluation, prompt engineering
- **Multi-Agent Systems**: Agent coordination, task delegation
- **API Integration**: OpenAI, Groq, Google APIs, custom services
- **UI/UX Development**: Gradio interfaces, interactive dashboards
- **Data Processing**: Text analysis, vector databases, content extraction
- **SEO Optimization**: Content analysis, keyword research, ranking factors

---

## Project Organization Notes

### Duplicate Projects Identified
- **SecurityHundley & Cybersecurity**: Both appear to be the same cybersecurity project. The polished version (2025-04_BaseModel-Chat-Bridge.ipynb) is recommended for showcase.
- **HR Chatbot Variants**: Multiple versions exist (2023-03_HR_RAG_Chatbot.ipynb, 2023-03_HR_RAG_Chatbot_polished.ipynb). The polished version is recommended.

### Recommended Showcase Focus
For potential employers, focus on these key projects:
1. **HR Assistant with RAG** - Demonstrates practical business application
2. **Base Model Chat Bridge** - Shows innovative model architecture techniques
3. **Multi-Agent RAG System** - Advanced AI architecture skills
4. **SEO Analysis Tool** - Complete product development cycle
5. **RAG Basics** - Educational value and clear explanations

---

**Contact:**  
For questions or feedback, reach out at kutyadog@gmail.com (Chris Johnson).
