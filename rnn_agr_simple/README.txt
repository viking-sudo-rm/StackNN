Simplified versions of subject-agreement dependencies used in Linzen, Dupoux and Goldberg (2016) -- do not contain the predictors derived from the parses of the sentences (see https://github.com/TalLinzen/rnn_agreement for more information).

numpred.train, numpred.valid and numpred.test: with a 9/1/90 train/dev/test split.

numpred.test.{0,1,2,3,4,5}: dependencies of the test set with 0 to 5 agreement attractors (nouns of the opposite number from the subject the occur between the subject and the verb). Only dependencies in which all of the nouns between the subject and the verb were of the same number (dependencies with "homogeneous intervention") are included.

Citation:

@article{linzen2016assessing,
    Author = {Linzen, Tal and Dupoux, Emmanuel and Goldberg, Yoav},
    Journal = {Transactions of the Association for Computational Linguistics},
    Title = {Assessing the ability of {LSTMs} to learn syntax-sensitive dependencies},
    Volume = {4},
    Pages = {521--535},
    Year = {2016}
}
