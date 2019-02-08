Feb 8,
I (noah) am running the alphabet_test_config reverse tasks with 10 symbols and a read size of 4
on the three interesting controller/struct pairs and saving the results in here


python run.py alphabet_test_config --controller LSTMSimpleStructController --struct Stack --model VanillaModel --savepath "saved_models/alphabet_exp2/10letters_lstm_stack"
python run.py alphabet_test_config --controller LinearSimpleStructController --struct Stack --model VanillaModel --savepath "saved_models/alphabet_exp2/10letters_stack"
python run.py alphabet_test_config --controller LSTMSimpleStructController --struct NullStruct --model VanillaModel --savepath "saved_models/alphabet_exp2/10letters_lstm"
