# Replication package for "Software Engineering and Foundation Models: Insights from Industry Blogs Using a Jury of Foundation Models"

This repository provides all the data and code required to reproduce our paper.

## Index for blog posts

Check the [posts_index](posts_index) folder for the index of blog posts:

- [FM4SE.md](posts_index%2FFM4SE.md)
- [SE4FM.md](posts_index%2FSE4FM.md)

## Prompts

- [prompts](prompts): contains the prompts we used for FM/LLM Jury:
    - [SE-FM_Area.txt](prompts%2FSE-FM_Area.txt): contains the SE-FM area classification prompt.
    - [FM4SE.txt](prompts%2FFM4SE.txt): contains the FM4SE activity classification prompt.
    - [SE4FM.txt](prompts%2FSE4FM.txt): contains the SE4FM activity classification prompt.

## Data

[./data](data) contains the data we used in our study:
- [company_blogs.json](data%2Fcompany_blogs.json): contains the list of company blogs we used in our study.
- [collected_blog_posts.csv](data%2Fcollected_blog_posts.csv): contains the metadata of 4,463 blog posts:
    - `id`: unique identifier of the blog post.
    - `title`: the title of the blog post.
    - `link`: the URL of the blog post.
    - `company`: the company that published the blog post.
    - `snippet`: the snippet of the blog post.
    - `area`: the SE-FM area of the blog post, i.e., FM4SE, SE4FM, and others.
- [fm4se_activities.csv](data%2Ffm4se_activities.csv): contains 155 FM4SE blog posts.
    - `activity`: the FM4SE activity.
    - `tasks`: the tasks related to the FM4SE activity.
    - other columns are the same as `collected_blog_posts.csv`.
- [se4fm_activities.csv](data%2Fse4fm_activities.csv): contains 997 SE4FM blog posts.
    - `activity`: the SE4FM activity.
    - `tasks`: the tasks related to the SE4FM activity.
    - other columns are the same as `collected_blog_posts.csv`.
