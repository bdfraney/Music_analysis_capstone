# Audio Analysis and Genre Classification
by Blake Franey

---

### Executive Summary

One of the fundamental problems in audio processing is audio classification.  Extracting useful features from auido data and using those to place audio into classes is essential to a lot of applications - especially genre classification in music.  Using royalty-free music scraped from various websites I will use a neural network, primarily a convolutional neural network, to classify the songs into 6 separate genres: Techno, House, Indie-rock, Garage-rock, Dubstep, and Drum and Bass.  

I used a Selenium webdriver to scrape music from the [Free Music Archive](http://freemusicarchive.org/); a website that allows you to download thousands of royalty-free music files.  This was initially going to be my only data source until I realized that the genres weren't very reliable.  Unfortunately, most of the other websites were not webscraper friendly and downloading the free music wasn't as easy since the other sites didn't have a simple list of songs with download links next to them.  I Initially scraped over 2000 songs but after going through the files and deleting all the ones that didn't fit into any of my genres I was left with just over 800.  After downloading from other sources my final count was around 1400.

The goal was to build a Neural Network to classify these tracks into 6 separate genres, the target metric was accuracy.  Baseline accuracy was ~17% so improving on that was the goal. 

The final model I was able to successfully run had a validation accuracy of 52% withe loss steadily decreasing with the training set so there wasn't too much overfitting.Unfortunately this was only on a subset of the data. Parsing the whole songset proved to be a much bigger resource sink than I had thought and consequently I am still left with my 240 song dataset. I had attempted to use the multiprocessing library to accelerate the process. It took just under 4 hours (wit to go through the whole songset with 8 cores running at 100% but when I went to use the data, the model wasn't learning anything and did the same or worse than baseline. 

So as of 8/26/19:

My "final" model did ok, I was able to improve the initial accuracy and develop a dense NN that was a little more robust than the simpler ones. There are many ways that to improve this, some I'd like to explore are using the Mel-freq spectrograms themselves (not coefficients) in a Convolutional2D Neural Network. Convolutional1D layers could also be explored if I extract the right feature for those inputs. Additionally, I only included 1 'enhanced' feature that could be combined with other enhanced versions of the MFCCs or more to make the model more robust. My final saved accuracy was 52% which will hopefully be improved once I am able to extract more robust features on the full dataset.  

## Notebooks


1. [Scraping](./Notebooks/01_Selenium_scraping.ipynb)

2. [EDA with Mutagen Library](./Notebooks/02_EDA_Mutagen_ID3.ipynb)

3. [EDA with LibROSA](./Notebooks/03_EDA_LibROSA.ipynb)

4. [Chroma Model](./Notebooks/04_Chroma_model.ipynb)

5. [Feature Extraction](./Notebooks/05_Feature_Extraction.ipynb)

6. [Model Pt. 1](./Notebooks/06_Models_Pt1.ipynb)

7. [Model Pt 2](./Notebooks/07_Models_Pt2.ipynb)


### Python APIs

- SciPy 1.3.x

- Pandas 0.25.x

- NumPy 1.17.x

- Scikit-learn 0.21.x

- Matplotlib 3.1.x

- Seaborn 0.9.x

- LibROSA 0.7.x

- Mutagen 1.42.x

- Tensorflow 2.0.x

- Selenium 3.141.x


### Additional Info

- [LibROSA](https://librosa.github.io/librosa/)
- [Mutagen](https://mutagen.readthedocs.io/en/latest/index.html)
- [.py files](./Python_Files) contains functions all of the functions. 
- [Audio Content Analysis Lecture Series](https://www.audiocontentanalysis.org/) The author's repo for this class is also being updated with his own python scripts for computations and functions related to the course.
