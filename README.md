# Aura: Avantlink Universal Reporting Assistant 🔮

**Aura is a state-of-the-art AI assistant that transforms natural language questions into precise JSON queries for Avantlink's Enterprise Data Warehouse. It empowers users to interact with complex data intuitively, eliminating the need for technical expertise in query languages.**

---

## 💡 Overview

In the world of data-driven decision-making, the ability to quickly and easily access information is paramount. However, traditional data warehouses often require users to have a deep understanding of complex query languages and data structures. This creates a bottleneck, slowing down the pace of analysis and limiting access to valuable insights.

Aura solves this problem by providing a natural language interface to Avantlink's data warehouse. With Aura, any user—from a sales executive to a marketing analyst—can simply ask a question in plain English and receive the data they need in a structured, usable format. This democratizes data access, accelerates reporting, and unlocks the full potential of your data assets.

## 💼 Business Value

Aura is more than just a technical marvel; it's a strategic asset designed to deliver tangible business value. By bridging the gap between complex data and the people who need it, Aura drives efficiency, fosters innovation, and improves the bottom line.

*   **🚀 Accelerate Time-to-Insight:** Eliminate the bottleneck of manual query writing. Business users get the data they need in seconds, not hours or days, allowing them to make faster, more informed decisions.
*   **Democratize Data Access:** Empower non-technical stakeholders across all departments—from sales and marketing to product and support— to self-serve their data needs. This fosters a more data-literate culture and frees up your analytics team to focus on higher-value strategic initiatives.
*   **📉 Reduce Operational Costs:** Automate the repetitive task of generating routine reports. Aura acts as a tireless digital assistant, reducing the workload on data analysts and developers and lowering the overall cost of data operations.
*   **💡 Unlock New Opportunities:** When data is this accessible, curiosity thrives. Users are more likely to explore data, ask new questions, and uncover hidden patterns and opportunities that would have otherwise gone unnoticed.

## ✨ Key Features

*   🗣️ **Natural Language to JSON:** At its core, Aura is a sophisticated translation engine, converting conversational English into precise, machine-readable JSON queries.
*   🎯 **High Accuracy:** Fine-tuned on a custom dataset of Avantlink's data structures and query patterns, Aura understands the nuances of the business and delivers highly accurate results.
*   🧬 **Data Augmentation:** Aura's training data was enriched using advanced data augmentation techniques, making the model robust and resilient to variations in user input.
*   ⚡️ **Efficient and Scalable:** Built with modern, efficient techniques like PEFT, LoRA, and 4-bit quantization, Aura is designed to be both powerful and cost-effective to operate.

## 🛠️ Technical Stack

Aura is built on a foundation of cutting-edge machine learning technologies:

*   🤖 **Model:** The core of Aura is a fine-tuned **StarCoder** model, a powerful large language model for code.
*   ⚙️ **Fine-Tuning:** The model was trained using **Parameter-Efficient Fine-Tuning (PEFT)** with **Low-Rank Adaptation (LoRA)**, allowing for rapid and efficient adaptation to the custom dataset.
*   🧊 **Quantization:** To optimize performance and reduce memory usage, Aura employs **4-bit quantization**.
*   📦 **Data Processing:** The training pipeline leverages the **Hugging Face `datasets` library** for efficient data handling and preparation.
*   🧠 **Frameworks:** The project is built with **PyTorch** and **Transformers**.

## 🌊 The Aura Data Pipeline: A Two-Act Play

The creation of Aura's training data is not a simple script; it's a two-act play where data is the protagonist, and a powerful AI acts as both its mentor and its muse. The entire flow is orchestrated across two key notebooks, transforming raw, historical query logs into a rich, diverse, and robust dataset ready to teach a new AI.

---

#### **Act I: The Genesis (`dataprep.ipynb`)**

**Scene:** *A digital forge, where raw ore is purified and prepared.*

The story begins with a humble CSV file, `reports_saved.csv`. This file is the project's "raw ore"—a log of thousands of saved reports, each containing a human-friendly `name` and a complex, machine-generated JSON `definition`.

1.  **The Smelting:** The `dataprep.ipynb` notebook first loads this raw data. Its first order of business is purification. It recognizes that many reports, despite having different names, share the exact same JSON definition. To create a clean, unambiguous training set, it performs a clever deduplication, ensuring that only one copy of each unique JSON query proceeds.

2.  **The First Transformation:** The notebook then takes these unique JSON `definition` strings and sends them, one by one, to the OpenAI GPT-4 API. It uses a carefully engineered prompt (`Human Readable Query From JSON`) to ask the AI a simple but profound question: *"Describe this machine query in plain English."*

3.  **The First Artifact:** The AI's answers—the newly created natural language descriptions—are meticulously collected. The notebook then saves this crucial output, a table mapping each unique JSON query to its new human-readable description, into a file named `OpenAI_Json_query_output.csv`. This file is the final artifact of Act I, and it becomes the key that unlocks the next stage of the process.

---

#### **Act II: The Alchemy (`create_training_data.ipynb`)**

**Scene:** *An alchemist's workshop, where the purified data is transmuted into gold.*

The second act begins where the first left off, with the `create_training_data.ipynb` notebook picking up the `OpenAI_Json_query_output.csv` file.

1.  **The Second Transformation:** This notebook performs a second, even more creative, AI-powered step. It takes the human-readable descriptions generated in Act I and sends them *back* to GPT-4. This time, the prompt is different (`create_alternatives_prompt`). It asks the AI: *"Rewrite this query in a few different ways, using different words but keeping the original meaning."* This is the data augmentation step, designed to teach the final model to be flexible and understand a wide variety of user phrasing. The AI's creative alternatives are saved to a new file, `report_query_variations.csv`.

2.  **The Great Merge:** The notebook now holds three key pieces of information for each original query: the ground-truth JSON `code`, its AI-generated `output` description, and a list of AI-generated `alternatives`. It masterfully merges all of this information into a single, comprehensive dataset.

3.  **The Final Forging:** With all the pieces assembled, the notebook performs its final, crucial transformation. It "melts" the data, restructuring it into the classic `prompt` -> `completion` format required for fine-tuning a language model. It creates multiple training examples from each original query:
    *   (Natural Language Description) -> (JSON `code`)
    *   (Alternative Description 1) -> (JSON `code`)
    *   (Alternative Description 2) -> (JSON `code`)

4.  **Materialization:** This final, golden dataset, now perfectly structured and rich with variation, is saved as `data/training/training_data_edw2_all.csv`. As its very last step, the notebook converts this CSV into the highly efficient binary `.arrow` format, creating the final `train` and `test` splits that will be fed directly to the StarCoder model.

---

This two-act data pipeline is the heart of the Aura project. It's a system that uses AI to create better AI, transforming simple logs into a sophisticated and robust training curriculum.

## 📁 Project Structure

The repository is organized into the following key directories:

*   📊 `data/`: Contains all data used in the project, including source data, prompts, training data, and vector stores.
*   📓 `notebooks/`: A collection of Jupyter notebooks for data preparation, training data creation, and model inference.
*   🐍 `scripts/`: Key Python scripts for training the model (`star_coder_trainer.py`) and running inference.

## 🚀 Getting Started

To get started with Aura, you'll need to have Python and the required dependencies installed. The core of the project is the `star_coder_trainer.py` script, which handles the model training.

```bash
# 1. Install the required dependencies
# (You'll need to create a requirements.txt file for this)

# 2. Run the training script
python star_coder_trainer.py --model_path bigcode/starcoderplus --dataset_name <your_dataset> --output_dir ./aura_model
```

---

📫 This project was created to demonstrate the power of large language models to solve real-world business problems. For any questions or inquiries, please reach out.
