# Replication package for "Software Engineering and Foundation Models: Insights from Industry Blogs Using a Jury of Foundation Models"

This repository contains all the data and code to replicate the results presented in our paper. If you're interested in reproducing the results, jump directly to [4. 🔄 Reproduce](#4--reproduce).  

## Table of Contents
1. [🌐 Blog Post Index](#1--blog-post-index-)
2. [💬 Prompts](#2--prompts-)
3. [📊 Data](#3--data-)
4. [🔄 Reproduce](#4--reproduce)
5. [♻️ Reuse](#5--reuse)
6. [📄 Citation](#6--citation)

---

## 1. **🌐 Blog Post Index**  
Explore **categorized blog posts** that highlight the role of FMs and SE in real-world practices. 

Check the [posts_index](posts_index) folder for detailed indexes of relevant blog posts:

- **[FM4SE.md](posts_index%2FFM4SE.md)**: Blogs on **Foundation Models *for* Software Engineering (FM4SE)**.  
- **[SE4FM.md](posts_index%2FSE4FM.md)**: Blogs on **Software Engineering *for* Foundation Models (SE4FM)**.  

## 2. **💬 Prompts**  
Prompts used for the FM/LLM jury can be found in the [prompts](prompts) folder:

- **[SEFM_Area.txt](prompts%2FSEFM_Area.txt)**: Classifies blog posts into **SE-FM areas**.  
- **[FM4SE.txt](prompts%2FFM4SE.txt)**: Focuses on activities related to **FM4SE**.  
- **[SE4FM.txt](prompts%2FSE4FM.txt)**: Focuses on activities related to **SE4FM**.  

## 3. **📊 Data**
The [data](data) folder contains all the datasets used in our study, including:

- **[company_blogs.json](data%2Fcompany_blogs.json)**:
  A JSON file containing blog sites from various companies.

- **[collected_blog_posts.csv](data%2Fcollected_blog_posts.csv)**:  
  A CSV containing **4,463 blog posts** with key metadata:
  - `id`: Unique identifier of the blog post  
  - `title`: Title of the blog post  
  - `link`: URL of the blog post  
  - `company`: The company that published the blog post.
  - `snippet`: the snippet of the blog post (provided by Google Search)
  - `area`: Classification area (`FM4SE`, `SE4FM`, or `Others`)

- **[FM4SE_activities.csv](data%2FFM4SE_activities.csv)**:  
  Contains **155 FM4SE blog posts**, with details about:
  - `activity`: FM4SE activity  
  - `tasks`: Tasks related to the FM4SE activity  
  - *Other columns*: Same as in `collected_blog_posts.csv`

- **[SE4FM_activities.csv](data%2FSE4FM_activities.csv)**:  
  Contains **997 SE4FM blog posts**, with details about:
  - `activity`: SE4FM activity  
  - `tasks`: Tasks associated with the SE4FM activity  
  - *Other columns*: Same as in `collected_blog_posts.csv`

## 4. **🔄 Reproduce**

To replicate the results of our study, follow these steps using the provided **Docker image**.  

### Step 1: Build the Docker Image

You can build the Docker image from the included `Dockerfile`:

```bash
docker build -t fmse_blogs .
```

Alternatively, if you have the pre-built Docker image file (`fmse_blogs_image.tgz`), load it into your Docker environment:  

```bash
docker load -i fmse_blogs_image.tgz 
```

### Step 2: Run the Docker Container

Once the image is built or loaded, run the container to see the results as follows: 

```bash
docker run -it -v "${PWD}/output":"/app/output" fmse_blogs
```

## 5. **♻️ Reuse**

This repository can also be reused to collect and analyze blog posts from other companies. Follow these steps to adapt the code:  

1. **Update Blog Sources:**  
   Modify [data/company_blogs.json](data/company_blogs.json) to include blog URLs for the companies you wish to analyze.

2. **Search Blog Posts:**  
   Run the script [scripts/search_blogs.py](scripts/search_blogs.py) to search for blog posts. Note: You will need to configure the following environment variables for Google Search API:
   - `GOOGLE_SEARCH_API_KEY`  
   - `GOOGLE_SEARCH_ENGINE_ID`

3. **Download Blog Posts:**  
   Use [scripts/download_blogs.py](scripts/download_blogs.py) to fetch the blog posts based on search results.

4. **Analyze Blog Posts:**  
   Apply your models to analyze the blog posts using the prompts provided in the [prompts](prompts) folder.

5. **Generate Reports:**  
   Use [scripts/report_results.py](scripts/report_results.py) to generate results and insights from the analysis.

6. **Create Index Files:**  
   Run [scripts/generate_mds.py](scripts/generate_mds.py) to create Markdown indexes for the blog posts.

### Reusing the FM/LLM Jury  
If you only wish to reuse the **FM/LLM Jury**, you can directly integrate our module located in [src/jury](src/jury).

## 6. **📄 Citation**

If you find this replication package useful, please cite our paper using the following BibTeX entry:

```bibtex
@misc{li_fmjury_2024,
      title={Software Engineering and Foundation Models: Insights from Industry Blogs Using a Jury of Foundation Models}, 
      author={Hao Li and Cor-Paul Bezemer and Ahmed E. Hassan},
      year={2024},
      eprint={2410.09012},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2410.09012}, 
}
```
