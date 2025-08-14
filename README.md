# AI Notebooks of Interest

This repository showcases a curated selection of AI projects and experiments I have developed & worked with since March 2023. 
I am actively (Aug 2025) in the process of editing these. Please check back again for changes.

1. **Featured Projects** – AI projects I have built.
2. **Experiments** – exploratory code and prototype work
3. **Reference** – Projects that I am watching / testing

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


### 🌉 Finetune Tiny LLM w/Custom Data
**[Fine_tune_llm_custom_data.ipynb](https://github.com/kutyadog/ai_notebooks/blob/main/Fine_tune_llm_custom_data.ipynb)**  
This project is focused on fine-tuning a small open-source language model (Qwen/Qwen2.5-3B-Instruct) on custom data related to a super-hero that I quickly invented (I wanted something that I knew the model would have no prior knowledge of).

**Key Features:**
- Fine-tuning: The core process of adapting a pre-trained model to custom data.
- LoRA (Low-Rank Adaptation): Utilized for efficient fine-tuning.
- Custom Character Data: Fine-tuning on unique, invented character information.
- Small Open-Source Model: Using a specific, publicly available model (Qwen/Qwen2.5-3B-Instruct).
- Workflow: The project outlines a complete process from setup and data preparation to training, saving, and testing the fine-tuned model.

---

## Experiments

A collection of exploratory notebooks testing various AI/ML techniques:
Note: Not all of these are my code... (should be obvious or credited).
Note: Not all of these are edited/cleaned up yet. In process of editing them now (Aug 2025).

### Conversational AI & Chatbots
- **2023-03_HR_RAG_Chatbot_polished.ipynb** - HR-focused chatbot with RAG
- **2024-01_AgentChatGroupChatRAG.ipynb** - Multi-agent group chat system
- **2025-04_HybridModelChatBridge.ipynb** - Hybrid Model Architecture - Base model to chat interface bridge
- **2023-10_ChatbotSimpleRAG.ipynb** - Lightweight RAG chatbot prototype
- **2023-07_LangChainChatGPT.ipynb** - Early LangChain integration experiments
- **2023-08_CampingRawExploration.ipynb** - Generative AI exploration

### RAG & Document Processing
- **2024-08_AiRAGBasics.ipynb** - Step-by-step RAG implementation tutorial
- **2023-03_EmbeddingsDocxRAG.ipynb** - Document embedding and retrieval

### SEO & Content Analysis
- **2023-12_AI_SEO_POC.ipynb** - AI-driven SEO analysis tool
- **2023-12_AI_SEO_Tests.ipynb** - SEO optimization experiments

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

## Reference

- **Structured Input/Output**: 
   -  [Outlines](https://github.com/dottxt-ai/outlines) : Outlines guarantees structured outputs during generation — directly from any LLM.

- **Chat Frameworks**: 
   -  [HayStack](https://github.com/deepset-ai/haystack) : AI orchestration framework with advanced retrieval methods; it's best suited for building RAG, question answering, semantic search or conversational agent chatbots.
   
   

---

## Project Organization Notes

### Duplicate Projects Identified
- **SecurityHundley & Cybersecurity**: Same project. The polished version () is recommended for showcase.
- **HR Chatbot Variants**: Multiple versions exist (2023-03_HR_RAG_Chatbot.ipynb, 2023-03_HR_RAG_Chatbot_polished.ipynb). The polished version is recommended.

---

**Contact:**  
For questions or feedback, reach out at kutyadog@gmail.com (Chris Johnson).
