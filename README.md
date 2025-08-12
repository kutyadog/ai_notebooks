# AI Notebooks Portfolio

This repository showcases a curated selection of AI projects and experiments developed since March 2023. 
I am in the process of cleaning these up and posting them (Aug 2025). Please check back for more.
It's organized into three sections:

1. **Showcase** – polished, production-ready notebooks demonstrating key skills  
2. **Experiments** – exploratory code and prototype work  
3. **Archive** – retired or hidden notebooks

---

## Featured Projects

### 🌉 Base Model Chat Bridge
**[2025-04_HybridModelChatBridge.ipynb](https://github.com/kutyadog/ai_notebooks/blob/main/showcase/2025-04_HybridModelChatBridge.ipynb)**  
Proof-of-concept for attempting to use an instruct model as a 'bridge' or 'interpreter' between base model and user by reformating user questions for specialized base models.

**Key Features:**
- Base-to-instruct model architecture
- Multi-model conversation flow
- Interactive Gradio interface
- Confidence scoring system
- Performance metrics analysis
diff>


---

## All Showcase Projects
Note: Not all of these are edited/cleaned up yet. In process of editing them now (Aug 2025).
### Conversational AI & Chatbots
- **2023-03_HR_RAG_Chatbot_polished.ipynb** - HR-focused chatbot with RAG
- **2024-01_AgentChatGroupChatRAG.ipynb** - Multi-agent group chat system
- **2025-04_HybridModelChatBridge.ipynb** - Base model to chat interface bridge
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
Note: Not all of these are my code... (should be obvious or credited).

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
- **SecurityHundley & Cybersecurity**: Both appear to be the same cybersecurity project. The polished version () is recommended for showcase.
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
