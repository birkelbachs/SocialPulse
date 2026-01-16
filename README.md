#  SocialPulse: An Open-Source Subreddit Sensemaking Toolkit

## 📖 Overview  
**SocialPulse** is an interactive **Reddit analysis and visualization framework** for exploring discussion topics, user overlap, and content similarity across subreddits. It is designed for **computational social science research** and supports both single-subreddit and cross-subreddit analysis.

The system provides:

- **Topic Modeling:** BERTopic-based discovery of latent discussion themes  
- **Cross-Subreddit Comparison:** Topic overlap, shared links, and overlapping users  
- **Bot-Aware Analysis:** Optional filtering and comparison of human vs bot-driven content  
- **Interactive Visualizations:** Plotly-based charts embedded directly in the web UI  

SocialPulse emphasizes *interpretability* and *exploratory analysis*, enabling researchers to understand not just **what** topics emerge, but **how** communities intersect and differ.

This code accompanies ongoing research and demo work for ICWSM-style analyses.

## 📦 Installation  

This project consists of two components: 
- **SocialPulse** (the main Flask application)
- **BotBuster** (an optional bot-detection module that runs in a separate environment)

The steps below walk through setting up both.

### 🔁 Clone this repository locally:  

```bash
git clone https://github.com/birkelbachs/SocialPulse.git
```

### 🤖 BotBuster Setup

BotBuster is used to identify likely bot accounts during subreddit analysis. It must be run in a separate Conda environment: 

```bash
cd SocialPulse/BotBuster-Universe/BotBuster
conda create -n botbusterEnv python=3.9.7 pip ipykernel -y
conda activate botbusterEnv
pip install -U pip
pip install tqdm joblib matplotlib seaborn emoji icecream
conda install -c conda-forge -y numpy=1.21 pandas=1.3 scikit-learn=0.24.2
conda deactivate botbusterEnv
cd ../../Reddit-Topic-Explorer
```
Note: Botbuster is automatically invoked by the main application when bot filtering is enabled. You do not need to run it manually after setup.

### 🧪 Main Application Environment (SocialPulse)

The main SocialPulse application runs in its own Conda environment to avoid dependency conflicts and ensure cross-platform compatibility.

```bash
conda create -n socialpulseEnv python=3.10
conda activate socialpulseEnv
```

SocialPulse uses WeasyPrint for PDF report generation, which requires several native libraries on macOS:
```bash
conda install -c conda-forge -y \
  weasyprint \
  glib \
  pango \
  cairo \
  gdk-pixbuf \
  libffi \
  fontconfig
```

Install Python Dependencies:
```bash
pip install -r requirements.txt
```

## ⚡ Quick Start  

### 1. **Set Up API Credentials**  

Copy the example enviornment file with your credentials:

**Reddit API**
Create a Reddit app at https://www.reddit.com/prefs/apps and fill in the following environment variables:

```bash
CLIENT_ID=your_reddit_client_id
CLIENT_SECRET=your_reddit_client_secret
REDDIT_USERNAME=reddit_username
REDDIT_PASSWORD=reddit_password
USER_AGENT="socialpulse-analysis"
```
**OpenAI API**
Create an API key at https://platform.openai.com/account/api-keys and fill in the following environment variables: 
```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_ORG_ID=your_openai_api_org_id
OPENAI_PROJECT_ID=your_openai_api_proj_id
```

### 2. **Run the Flask App**  

```bash
python demo.py
```
The application will be available at: 
```bash
http://localhost:5000
```

### 3. **Analyze Subreddits**  

From the web interface: 
- Enter one or more subreddits (e.g., politics, conspiracy)
- Select analysis options (bot filtering, images, thresholds)
- Click **Analyze** to generate results

All plots and statistics are rendered dynamically in the browser.

## 🖼️ System Architecture
The pipeline below illustrates the SocialPulse analysis flow:
1. Reddit data ingestion (posts + comments)
2. Optional bot filtering and preprocessing 
3. Topic modeling with BERTopic
4. Cross-community similarity analysis
5. Interactive visualization and export report

Plots are generated server-side and injected into the frontend using Plotly HTML components.


## 🧩 Key Features  
- **Multi-Subreddit Analysis:** Compare communities side-by-side
- **Topic-Level Matching:** Identify semantically similar topics across subreddits
- **User Overlap Detection:** Highlight shared participants and potential bridges
- **Bot-Aware Controls:** Adjustable thresholds and comparison modes
- **Export-Ready Reports:** HTML templates compatible with PDF generation (WeasyPrint)


## 🤝 Contributing 

We welcome contributions! To get started:  

1. **Fork & Clone**  
   ```bash
    git clone https://github.com/birkelbachs/SocialPulse.git
    cd socialpulse 
    conda install -c conda-forge -y \
      weasyprint \
      glib \
      pango \
      cairo \
      gdk-pixbuf \
      libffi \
      fontconfig
    pip install -r requirements.txt   
   ```  

2. **Run Locally**  
   ```bash
    python demo.py
   ```  
   Make sure all tests pass before opening a pull request. 

3. **Submitting Changes**  
   - Open a pull request (PR) with a clear description of your changes.  
   - Include screenshots or output examples when modifying visualizations
   - If you are adding features or modifying behavior, update this README with relevant instructions.  

## 📝 Notes  
- SocialPulse is intended for research and exploratory analysis, not real-time monitoring.
- Topic modeling results depend on subreddit size, preprocessing, and BERTopic hyperparameters.
- Bot detection is heuristic-based and should be interpreted with caution.
- The framework is extensible to other platforms with minimal changes to the ingestion layer.

## 📚 Dependencies & Acknowledgements

This project makes use of **BotBuster**, a multi-platform bot detection framework, for identifying likely automated accounts in Reddit data.

If you use SocialPulse in academic work, please also cite BotBuster:

```bibtex
@inproceedings{ng2023botbuster,
  title={Botbuster: Multi-platform bot detection using a mixture of experts},
  author={Ng, Lynnette Hui Xian and Carley, Kathleen M},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  volume={17},
  pages={686--697},
  year={2023}
}
```

## ✨ Citation

If you use this code, please use the following citation:

```bibtex
@inproceedings{birkelbach2025socialpulse,
  title={SocialPulse: An Open-Source Subreddit Sensemaking Toolkit},
  author={Birkelbach, Stephanie and Teleki, Maria and Carragher, Peter and Dong, Xiangjue and Bhatnagar, Nehul and Caverlee, James},
  year={2025},
}
```
