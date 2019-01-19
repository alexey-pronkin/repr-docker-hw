#!/bin/bash
cd code && python IMBD.py;
cd ../latex && pdflatex paper.tex && cp paper.pdf ../results/;