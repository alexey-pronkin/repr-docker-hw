#!/bin/bash
python IMDB.py;
cd ../latex && pdflatex paper.tex && cp paper.pdf ../results/;