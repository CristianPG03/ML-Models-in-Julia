#________________________________________________________________________________________________________________#
#________________________________________________________________________________________________________________#
#                             Document for load the packages needed in the project                               #
#________________________________________________________________________________________________________________#
#________________________________________________________________________________________________________________#



using DelimitedFiles;
using Random;
using Statistics;
using Flux.Losses;
using Flux;
using Pkg;
using ScikitLearn;
using JLD2;
using PyCallJLD;
using BSON: @save;
using FileIO;
using Plots;
using StatsBase;
using Base;



@sk_import neighbors: KNeighborsClassifier;
@sk_import ensemble:(AdaBoostClassifier, GradientBoostingClassifier)
@sk_import svm:SVC
@sk_import ensemble:BaggingClassifier
@sk_import ensemble:StackingClassifier
@sk_import ensemble:VotingClassifier
@sk_import tree:DecisionTreeClassifier
@sk_import linear_model:LogisticRegression
@sk_import naive_bayes:GaussianNB 
@sk_import decomposition:PCA