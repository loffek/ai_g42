# AI project ai\_g42

### SETUP

To install dependencies:

	$ pip install -r requirements.txt

*OBS* Please update requirements.txt after adding dependencies


Before anything works, you will need run the setup file

	$ source setup

to set the environment variable "NLTK\_DATA"



### Usage

#### To train models:

	$ ./trainer.py -k 1 -s laplace -f savefiles/somefile -p data/corpora/posrev_train.txt -n data/corpora/negrev_train.txt

(will create/overwrite the file "savefiles/somefile")


#### To classify a review using trained models:

	$ ./classifier.py -f savefiles/somefile

This will wait for input from stdin. Example:


	$ ./classifier.py -f savefile
	"This movie sucks balls"
	NEGATIVE

or:

	$ echo "This movie sucks balls" | ./classifier.py -f savefile
	NEGATIVE


#### To make a full test of the trained models:

	./tester.py -f savefiles/somefile -p data/corpora/posrev_test.txt -n data/corpora/negrev_test.txt
