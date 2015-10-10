# AI project ai_g42

### SETUP

To install dependencies:

	$ pip install -r requirements.txt

*OBS* Please update requirements.txt after adding dependencies


### Usage

To train models:

	$ ./trainer.py -k 1 -s laplace -f savefile -p data/posrev.txt --n data/negrev.txt

(will create/overwrite the file "savefile")


To classify a review using trained models:

	$ ./classifier.py -f savefile

This will wait for input from stdin. Example:


	$ ./classifier.py -f savefile
	"This movie sucks balls"
	NEGATIVE

or:

	$ echo "This movie sucks balls" | ./classifier.py -f savefile
	NEGATIVE

