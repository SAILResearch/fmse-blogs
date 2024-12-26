# Replication package for "Software Engineering and Foundation Models: Insights from Industry Blogs Using a Jury of Foundation Models"

This repository provides all the data and code related to our paper.
Jump to [4. 🔄 Reproduce](#4--reproduce) if you want to reproduce the results. 

## Table of Contents
1. [🌐 Blog Post Index](#1--blog-post-index-)
2. [💬 Prompts](#2--prompts-)
3. [📊 Data](#3--data-)
4. [🔄 Reproduce](#4--reproduce)

## 1. **🌐 Blog Post Index**  
Explore **categorized blog posts** that highlight the role of FMs and SE in real-world practices. 

Check the [`posts_index`](posts_index) folder for detailed indexes of relevant blog posts:

- **[FM4SE.md](posts_index%2FFM4SE.md)**: Blogs on **Foundation Models *for* Software Engineering (FM4SE)**.  
- **[SE4FM.md](posts_index%2FSE4FM.md)**: Blogs on **Software Engineering *for* Foundation Models (SE4FM)**.  

## 2. **💬 Prompts**  
Prompts used for the FM/LLM jury:

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
  - `area`: Classification area (`FM4SE`, `SE4FM`, or others)

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

The `Dockerfile` in this directory has been used to create the docker image `fmse_blogs_image.tgz` using this command:

```bash
docker build -t fmse_blogs .
```

If you have downloaded the `fmse_blogs_image.tgz` file, you can load it into your Docker environment using the following command in your terminal, from the directory containing the `.tgz` file:

```bash
docker load -i fmse_blogs_image.tgz 
```

and run via 
```
docker run -it -v"${PWD}/output":"/app/output" fmse_blogs
```