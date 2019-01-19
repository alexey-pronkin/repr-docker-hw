# Base image
FROM python:3.7

# Author information
MAINTAINER pronkin.aleskey@skoltech.ru


# Install latex.
RUN apt-get update && apt-get install -y texlive

# Install necessary libraries
RUN pip install numpy scipy matplotlib sklearn lightgbm pandas jupyter tqdm

# Add necessary files. Good practice to do it at the end
# in order to avoid reinstallation of dependencies when files change
ADD data ./data
ADD latex ./latex
ADD run.sh ./
ADD IMDB.py ./

# Make run.sh executable
RUN chmod +x run.sh

# /example/results contents will be shared with the host
# if -v option used with "docker run" command
VOLUME /results

CMD ./run.sh