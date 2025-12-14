# AI Tool Usage Disclosure

## Tools Used
- ChatGPT – used occasionally for explanations of concepts (e.g. autoencoders, HMM/GMM), debugging guidance, and help understanding error messages.
- Claude – used for help with documentation wording, code cleanup suggestions, and organizing project structure ideas.
- Windsurf AI assistant – used to scaffold some parts of the project structure and suggest improvements to code organization.
- GitHub Copilot for autocomplete

## Significant Contributions
- AI tools were used to **suggest** code snippets, structures, and refactorings for parts of the project (e.g. data loading, preprocessing, model structure, tests, and configuration files).  
- In all cases, I **reviewed, adapted, and integrated** these suggestions myself. I am responsible for the final code, decisions, and architecture.  
- AI assistance was **supportive and partial**, not determinative: it did not replace my own work or understanding of the project.

## Learning Moments
- Used AI to better understand how autoencoders, GMMs, and HMMs can be applied to market regime detection.  
- Improved my understanding of Python best practices (project structure, type hints, tests, configuration files).  
- Learned how to write clearer documentation and commit messages with the help of AI-generated examples.

## Detailed AI Assistance

Throughout the project, AI tools provided targeted support on specific elements of the workflow. Below are the main areas where AI assistance was useful, with concrete examples.

### Data Loading & Preprocessing
- Suggested ways to structure the `yfinance` data loader.  
- Helped identify typical preprocessing steps for financial time series (e.g., handling missing timestamps, normalization strategies, feature scaling).  
- Provided examples of how to structure a clean preprocessing function and separate raw vs processed data.

### Model Architecture & Training
- Offered suggestions for designing the autoencoder architecture (layer sizes, activation choices, reconstruction loss).  
- Helped clarify how dimensionality reduction through a latent space can support market regime detection.  
- Provided small code snippets illustrating training loops, optimizer configuration, and basic training patterns (which I adapted myself).

### Probabilistic Models (GMM / HMM)
- Explained how Gaussian Mixture Models and Hidden Markov Models can be combined with the latent representation.  
- Suggested typical parameter choices (e.g., number of components, covariance types).  
- Provided conceptual explanations of how regime detection works in practice on financial time series.

### Evaluation & Visualizations
- Helped brainstorm ways to visualize reconstruction error, latent space clusters, and time-coloured regimes.  
- Suggested plotting patterns (subplot layouts, axis labels, legends) to make results more interpretable.

### Project Organization
- Provided example project layouts inspired by common data-science templates (e.g., separating `src/`, `configs/`, `data/raw`, `data/processed`, `notebooks/`).  
- Helped refine file naming conventions and clarify the responsibilities of each folder.

### Documentation & Communication
- Helped rephrase documentation sections for clarity and conciseness.  
- Provided examples of clean commit messages and README sections.  
- Assisted in structuring some of the explanations used in the written report and oral/video presentation.
