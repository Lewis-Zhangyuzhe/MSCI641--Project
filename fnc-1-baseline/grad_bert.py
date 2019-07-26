import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats, bert_body_features, bert_headline_features
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from bert_serving.client import BertClient
from utils.system import parse_params, check_version
import pandas as pd


def generate_features(stances,dataset,name, all_data=None):
	h, b, y = [],[],[]

	for stance in stances:
		y.append(LABELS.index(stance['Stance']))
		h.append(stance['Headline'])
		b.append(dataset.articles[stance['Body ID']])

	X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
	X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
	X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
	X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
    # bc = BertClient()
    # X_body = gen_or_load_feats(bert_body_features, h, b, "features/bert_body."+name+".npy", bc)
    # X_headline = gen_or_load_feats(bert_headline_features, h, b, "features/bert_headline."+name+".npy", bc)
    # X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, X_body, X_headline]
	if name == "competition":
		pair = np.load("./features/test_pair.npy")
	else:
		# bodies = np.load("./features/bert_body.train.npy")
		# headlines = np.load("./features/bert_headline.train.npy")
		pair = np.load("./features/train_pair.npy")
	bodies_index = []
	for stance in stances:
		bodies_index.append(all_data.stances.index(stance))
		# X_bodies = bodies[np.array(bodies_index)]    	
		# X_headlines = headlines[np.array(bodies_index)]
	X_pair = pair[bodies_index]
	X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, X_pair]
	return X,y


if __name__ == "__main__":
	check_version()
	parse_params()

	#Load the training dataset and generate folds
	d = DataSet()
	folds,hold_out = kfold_split(d,n_folds=10)
	fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

	# Load the competition dataset
	competition_dataset = DataSet("competition_test")
	X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition", competition_dataset)

	Xs = dict()
	ys = dict()

	# Load/Precompute all features now
	X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout", d)
	for fold in fold_stances:
		
		Xs[fold], ys[fold] = generate_features(fold_stances[fold],d,str(fold), d)


	best_score = 0
	best_fold = None


	# Classifier for each fold
	for fold in fold_stances:
	    ids = list(range(len(folds)))
	    del ids[fold]

	    X_train = np.vstack(tuple([Xs[i] for i in ids]))
	    y_train = np.hstack(tuple([ys[i] for i in ids]))

	    X_test = Xs[fold]
	    y_test = ys[fold]

	    clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
	    clf.fit(X_train, y_train)

	    predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
	    actual = [LABELS[int(a)] for a in y_test]

	    fold_score, _ = score_submission(actual, predicted)
	    max_fold_score, _ = score_submission(actual, actual)

	    score = fold_score/max_fold_score

	    print("Score for fold "+ str(fold) + " was - " + str(score))
	    if score > best_score:
	        best_score = score
	        best_fold = clf



	#Run on Holdout set and report the final score on the holdout set
	predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
	actual = [LABELS[int(a)] for a in y_holdout]

	print("Scores on the dev set")
	report_score(actual,predicted)
	print("")
	print("")

	# Run on competition dataset
	predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
	actual = [LABELS[int(a)] for a in y_competition]

	print("Scores on the test set")
	report_score(actual,predicted)
	print("")
	print("")

	stance_list = predicted
	bodyid_list = []
	headline_list = []
	for i in range(len(competition_dataset.stances)):
		bodyid_list.append(competition_dataset.stances[i]['Body ID'])
		headline_list.append(competition_dataset.stances[i]['Headline'])
	df = pd.DataFrame(data={'Headline': headline_list, 'Body ID': bodyid_list, "Stance": stance_list})
	df.to_csv("gra_bert_pair_submission.csv", index=False, encoding="utf-8")