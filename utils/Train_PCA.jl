#________________________________________________________________________________________________________________#
#________________________________________________________________________________________________________________#
#               Functions for train the models (Using PCA technique for reduce the data)                         #
#________________________________________________________________________________________________________________#
#________________________________________________________________________________________________________________#





#Same function as developed in the Units using PCA for reduce the dimensionality of the data
function modelCrossValidation_PCA(modelType::Symbol,
    modelHyperparameters::Dict,
    inputs::AbstractArray{<:Real,2},
    targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1})

        k = maximum(crossValidationIndices);
        f_score = Vector(undef, k);
        acc = Vector(undef, k);


    # Artificial Neural Network
    if (modelType == :ANN)

        #Converting the targets into a boolean matrix 
        targets = oneHotEncoding(targets);

        n = size(inputs,1)

        #Call the function of the previous units that train the model using crossvalidation
        model = trainClassANN_PCA(modelHyperparameters["topology"], (inputs, targets), crossValidationIndices; 
            transferFunctions = modelHyperparameters["transferFunctions"], maxEpochs = 1000, minLoss = 0.0,
            learningRate = modelHyperparameters["learningRate"], repetitionsTraining = modelHyperparameters["repetitionsTraining"],
            validationRatio = modelHyperparameters["validationRatio"], maxEpochsVal = modelHyperparameters["maxEpochsVal"])

        
    elseif (modelType == :SVM)

        
        #Define a loop and for each iteration is a train for each fold in the cross-validation
        for i in 1:k

            #Defining the model
            model = SVC(kernel = modelHyperparameters["kernel"], 
            degree = modelHyperparameters["kernelDegree"], 
            gamma = modelHyperparameters["kernelGamma"], 
            C = modelHyperparameters["C"]);

            kernel = modelHyperparameters["kernel"];
            c = modelHyperparameters["C"];
            gamma = modelHyperparameters["kernelGamma"];
            degree = modelHyperparameters["kernelDegree"];

            folder_path = "utils/ModelsTrained/modelsmodelsvm $kernel, $c, $gamma, $degree";

            # Create the folder if it doesn't exist
            if !isdir(folder_path)
                mkpath(folder_path)
            end

            #Check if there are enough instances for each subset
            @assert((sum(crossValidationIndices .!= i)) > 1); 
        
            #Define a variable with the number of elements of the k subset 
            nEl = sum(crossValidationIndices .== i);
            n = size(inputs,1);

            
            println(" ");
            println("------------------------------------------------------------------------------------------------------------------");
            println("-------------------------------------------------- FOLD: ", i, " -------------------------------------------------------");
            println("------------------------------------------------------------------------------------------------------------------");
        
            #Define the train and test subsets
            inputs_training = inputs[(crossValidationIndices.!=i),:];
            targets_training = reshape(targets[(crossValidationIndices.!=i),:], sum(crossValidationIndices.!=i));
            inputs_test = inputs[(crossValidationIndices.==i),:];
            targets_test = reshape(targets[(crossValidationIndices.==i),:], sum(crossValidationIndices.==i));

            pca_95 = PCA(0.95)
            ScikitLearn.fit!(pca_95,inputs_training);

            inputs_training_pca = pca_95.transform(inputs_training);
            inputs_test_pca = pca_95.transform(inputs_test);

            println("Size: ", size(inputs_training), " -> ", size(inputs_training_pca));
            siz = Vector(undef,2);
            siz[1] = size(inputs_training);
            siz[2] =  size(inputs_training_pca);

            size_file_path = joinpath(folder_path, "size$i.jld2");
            JLD2.save(size_file_path,"siz", siz);
        
        

            #Training the model
            ScikitLearn.fit!(model, inputs_training_pca, targets_training);


            #Print of the results
            outputs_model = ScikitLearn.predict(model, inputs_test_pca);


            model1_file_path = joinpath(folder_path, "targets_svmmodel$i.jld2");
            JLD2.save(model1_file_path,"targets_test",targets_test);

            
            model2_file_path = joinpath(folder_path, "outputs_svmmodel$i.jld2");
            JLD2.save(model2_file_path,"outputs_model",outputs_model);


            #Comparing and printing the outputs
            metric_svm = confusionMatrix(outputs_model, targets_test);
            
            println(" ");
            println("Metrics: ");
            # println("Confusion Matrix: ", metric_svm[1]);
            println("Weighted sensivity: ", metric_svm[2]);
            println("Weighted specificity: ", metric_svm[3]);
            println("Weighted precision: ", metric_svm[4]);
            println("Weighted F-score: ", metric_svm[6]);
            println("Accuracy: ", metric_svm[12]);

            f_score[i] = metric_svm[6];
            acc[i] = metric_svm[12]


        end 
        
        println("")
        println("The mean accuracy of all folds is: ", mean(acc), " with a standard deviation of: ", std(acc));
        println("The mean F1-score of all folds is: ", mean(f_score), " with a standard deviation of: ", std(f_score));


    #KNN
    elseif (modelType == :kNN)


        #Define a loop and for each iteration is a train for each fold in the cross-validation
        for i in 1:k

            #Defining the model
            model = KNeighborsClassifier(n_neighbors = modelHyperparameters["n_neighbors"]);

            k = modelHyperparameters["n_neighbors"]
            
            folder_path = "utils/ModelsTrained/modelknn $k";

            # Create the folder if it doesn't exist
            if !isdir(folder_path)
                mkpath(folder_path)
            end

            #Check if there are enough instances for each subset
            @assert((sum(crossValidationIndices .!= i)) > 1); 
        
            #Define a variable with the number of elements of the k subset 
            nEl = sum(crossValidationIndices .== i);
            n = size(inputs,1);


            
            println(" ");
            println("------------------------------------------------------------------------------------------------------------------");
            println("-------------------------------------------------- FOLD: ", i, " -------------------------------------------------------");
            println("------------------------------------------------------------------------------------------------------------------");



            #Define the train, test subsets 
            inputs_training = inputs[(crossValidationIndices.!=i),:];
            targets_training = reshape(targets[(crossValidationIndices.!=i),:], sum(crossValidationIndices.!=i));
            inputs_test = inputs[(crossValidationIndices.==i),:];
            targets_test = reshape(targets[(crossValidationIndices.==i),:], sum(crossValidationIndices.==i));

            pca_95 = PCA(0.95)
            ScikitLearn.fit!(pca_95,inputs_training);

            inputs_training_pca = pca_95.transform(inputs_training);
            inputs_test_pca = pca_95.transform(inputs_test);

            println("Size: ", size(inputs_training), " -> ", size(inputs_training_pca));
            siz = Vector(undef,2);
            siz[1] = size(inputs_training);
            siz[2] =  size(inputs_training_pca);

            size_file_path = joinpath(folder_path, "size$i.jld2");
            JLD2.save(size_file_path,"siz", siz);
        

            #Trainig the model
            ScikitLearn.fit!(model, inputs_training_pca, targets_training);
            
            #Print the results
            outputs_model = ScikitLearn.predict(model, inputs_test_pca);


            model1_file_path = joinpath(folder_path, "targets_knnmodel$i.jld2");
            JLD2.save(model1_file_path,"targets_test",targets_test);

            
            model2_file_path = joinpath(folder_path, "outputs_knnmodel$i.jld2");
            JLD2.save(model2_file_path,"outputs_model",outputs_model);
        
            metric_knn = confusionMatrix(outputs_model, targets_test);
        
            println(" ");
            println("Metrics: ");
            # println("Confusion Matrix: ", metric_knn[1]);
            println("Weighted sensivity: ", metric_knn[2]);
            println("Weighted specificity: ", metric_knn[3]);
            println("Weighted precision: ", metric_knn[4]);
            println("Weighted F-score: ", metric_knn[6]);
            println("Accuracy: ", metric_knn[12]);
            f_score[i] = metric_knn[6];
            acc[i] = metric_knn[12]


        end 

        println("")
        println("The mean accuracy of all folds is: ", mean(acc), " with a standard deviation of: ", std(acc));
        println("The mean F1-score of all folds is: ", mean(f_score), " with a standard deviation of: ", std(f_score));

    elseif (modelType == :DecisionTree)

        
        #Define a loop and for each iteration is a train for each fold in the cross-validation
        for i in 1:k
            

            #Defining the model
            model = DecisionTreeClassifier(max_depth = modelHyperparameters["max_depth"], random_state = modelHyperparameters["random_state"]);

            d = modelHyperparameters["max_depth"];

            folder_path = "utils/ModelsTrained/modeldt $d";

            # Create the folder if it doesn't exist
            if !isdir(folder_path)
                mkpath(folder_path)
            end


            #Check if there are enough instances for each subset
            @assert((sum(crossValidationIndices .!= i)) > 1); 
        
            #Define a variable with the number of elements of the k subset 
            nEl = sum(crossValidationIndices .== i);
            n = size(inputs,1);

            
            println(" ");
            println("------------------------------------------------------------------------------------------------------------------");
            println("-------------------------------------------------- FOLD: ", i, " -------------------------------------------------------");
            println("------------------------------------------------------------------------------------------------------------------");


        
            #Define the train, test subsets 
            inputs_training = inputs[(crossValidationIndices.!=i),:];
            targets_training = reshape(targets[(crossValidationIndices.!=i),:], sum(crossValidationIndices.!=i));
            inputs_test = inputs[(crossValidationIndices.==i),:];
            targets_test = reshape(targets[(crossValidationIndices.==i),:], sum(crossValidationIndices.==i));

            pca_95 = PCA(0.95)
            ScikitLearn.fit!(pca_95,inputs_training);
    
            inputs_training_pca = pca_95.transform(inputs_training);
            inputs_test_pca = pca_95.transform(inputs_test);
    
            println("Size: ", size(inputs_training), " -> ", size(inputs_training_pca));
            siz = Vector(undef,2);
            siz[1] = size(inputs_training);
            siz[2] =  size(inputs_training_pca);
    
            size_file_path = joinpath(folder_path, "size$i.jld2");
            JLD2.save(size_file_path,"siz", siz); 

            #Training the model
            ScikitLearn.fit!(model, inputs_training_pca, targets_training);

            #Print the results
            outputs_model = ScikitLearn.predict(model, inputs_test_pca);

            model1_file_path = joinpath(folder_path, "targets_dtmodel$i.jld2");
            JLD2.save(model1_file_path,"targets_test",targets_test);

            
            model2_file_path = joinpath(folder_path, "outputs_dtmodel$i.jld2");
            JLD2.save(model2_file_path,"outputs_model",outputs_model);

            metric_dt = confusionMatrix(outputs_model, targets_test);
                
            println(" ");
            println("Metrics: ");
            # println("Confusion Matrix: ", metric_dt[1]);
            println("Weighted sensivity: ", metric_dt[2]);
            println("Weighted specificity: ", metric_dt[3]);
            println("Weighted precision: ", metric_dt[4]);
            println("Weighted F-score: ", metric_dt[6]);
            println("Accuracy: ", metric_dt[12]);

            f_score[i] = metric_dt[6];
            acc[i] = metric_dt[12]


        end 

        println("")
        println("The mean accuracy of all folds is: ", mean(acc), " with a standard deviation of: ", std(acc));
        println("The mean F1-score of all folds is: ", mean(f_score), " with a standard deviation of: ", std(f_score));
    else
        print("ERROR IN MODEL TYPE")
    end
end

#Same function as developed in the Units using PCA for reduce the dimensionality of the data
function trainClassANN_PCA(topology::AbstractArray{<:Int,1}, 
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, 
    kFoldIndices::Array{Int64,1}; 
    transferFunctions::AbstractArray{<:Function,1}=fill(sigmoid, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, repetitionsTraining::Int=1, 
    validationRatio::Real=0.0, maxEpochsVal::Int=20)

    #Defining the train dataset
    (inputs, targets) = trainingDataset;
    #Check if the sizes of the inputs and the targets it is the same
    @assert(size(inputs,1)==size(targets,1));
    k = maximum(kFoldIndices);
    folder_path = "utils/ModelsTrained/modelann $topology, $transferFunctions";

    # Create the folder if it doesn't exist
    if !isdir(folder_path)
        mkpath(folder_path)
    end
    
    f_score = Vector(undef,k);
    acc = Vector(undef,k);


    for i in 1:k

        #Check if there are enough instances for each subset
        @assert((sum(kFoldIndices .!= i)) > 1); 

        #Define a variable with the number of elements of the k subset 
        nEl = sum(kFoldIndices .== i);
        n = size(inputs,1);

        println(" ");
        println("------------------------------------------------------------------------------------------------------------------");
        println("-------------------------------------------------- FOLD: ", i, " -------------------------------------------------------");
        println("------------------------------------------------------------------------------------------------------------------");


        #Define the train, test and validation subsets (the validation and input subsets defined with the HoldOut function)
        inputs_tv = inputs[(kFoldIndices.!=i),:];
        targets_tv = targets[(kFoldIndices.!=i),:];
        train_ind, val_ind = holdOut( (n - nEl), 0.2)

        inputs_training = inputs_tv[train_ind , :];
        targets_training = targets_tv[train_ind , :];

        inputs_validation = inputs_tv[val_ind , :];
        targets_validation = targets_tv[val_ind ,:];
        
        inputs_test = inputs[(kFoldIndices.==i),:];
        targets_test = targets[(kFoldIndices.==i),:];

        pca_95 = PCA(0.95)
        ScikitLearn.fit!(pca_95,inputs_training);

        inputs_training_pca = pca_95.transform(inputs_training);
        inputs_validation_pca = pca_95.transform(inputs_validation);
        inputs_test_pca = pca_95.transform(inputs_test);

        println("Size: ", size(inputs_training), " -> ", size(inputs_training_pca));
        siz = Vector(undef,2);
        siz[1] = size(inputs_training);
        siz[2] =  size(inputs_training_pca);

        size_file_path = joinpath(folder_path, "size$i.jld2");
        JLD2.save(size_file_path,"siz", siz);

        

        #Call the previous train function
        (ann, annTrainLosses, annValLosses, annTestLosses) = trainClassANN(topology, (inputs_training_pca, targets_training), 
        (inputs_validation_pca, targets_validation),(inputs_test_pca, targets_test) , transferFunctions, maxEpochs, minLoss, learningRate, maxEpochsVal);


        #Saving the outputs of the model
        outputs_model = ann(inputs_test_pca');
        outputs_model = outputs_model'

        model1_file_path = joinpath(folder_path, "targets_annmodel$i.jld2");
        JLD2.save(model1_file_path,"targets_test",targets_test);

        
        model2_file_path = joinpath(folder_path, "outputs_annmodel$i.jld2");
        JLD2.save(model2_file_path,"outputs_model",outputs_model);



        
        #Comparing the actual outputs with the real ouputs (targets)
        metric_ann = confusionMatrix(outputs_model, targets_test);
        
        #Print the results
        println(" ");
        println("Metrics: ");
        # println("Confusion Matrix: ", metric_ann[1]);
        println("Weighted sensivity: ", metric_ann[2]);
        println("Weighted specificity: ", metric_ann[3]);
        println("Weighted precision: ", metric_ann[4]);
        println("Weighted F-score: ", metric_ann[6]);
        println("Accuracy: ", metric_ann[12]);
        f_score[i] = metric_ann[6];
        acc[i] = metric_ann[12]


    end 

    println("")
    println("The mean accuracy of all folds is: ", mean(acc), " with a standard deviation of: ", std(acc));
    println("The mean F1-score of all folds is: ", mean(f_score), " with a standard deviation of: ", std(f_score));
end

#Same function as developed in the Units using PCA for reduce the dimensionality of the data
function trainClassEnsemble_PCA(estimators::AbstractArray{Symbol,1}, 
    modelsHyperParameters:: AbstractArray{Dict, 1},     
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},    
    kFoldIndices:: Array{Int64,1})

    #Defining the datasets
    (inputs, targets) = trainingDataset;

    #Counting the number of the folds
    k = maximum(kFoldIndices); 

    #Initializing the vectors for the metrics
    ens_acc = Vector{Float64}(undef, k);
    ens_f1 = Vector{Float64}(undef, k);
    folder_path = "utils/ModelsTrained/modelensemble";

    if !isdir(folder_path)
        mkdir(folder_path)
    end

    #Loop iterating the folds
    for i in 1:k


        #Initializing a dictionary for the models (the models are new in each fold)
        models_dict = Dict()

        println("__________________________________________________________________");
        println("_______________________________$i Fold_____________________________");
        println("__________________________________________________________________");

        #Define the train, test subsets
        inputs_training = inputs[(kFoldIndices.!=i),:];
        targets_training = reshape(targets[(kFoldIndices.!=i),:], sum(kFoldIndices.!=i));
        inputs_test = inputs[(kFoldIndices.==i),:];
        targets_test = reshape(targets[(kFoldIndices.==i),:], sum(kFoldIndices.==i));

        pca_95 = PCA(0.95)
        ScikitLearn.fit!(pca_95,inputs_training);

        inputs_training_pca = pca_95.transform(inputs_training);
        inputs_test_pca = pca_95.transform(inputs_test);

        println("Size: ", size(inputs_training), " -> ", size(inputs_training_pca));
        siz = Vector(undef,2);
        siz[1] = size(inputs_training);
        siz[2] =  size(inputs_training_pca);

        size_file_path = joinpath(folder_path, "size$i.jld2");
        JLD2.save(size_file_path,"siz", siz); 


        #Checking if every subset have more than one element
        @assert((sum(kFoldIndices .!= i)) > 1); 

        #Total number of elements and number of elements in the subset k
        nEl = sum(kFoldIndices .== i);
        n = size(inputs,1);
        num_models = length(estimators)
        
        #Iterating in each model
        for j in 1:num_models
            #If the model is a SVM
            if modelsHyperParameters[j]["model"] == "SVM"

                #Defining the model
                model_SVM = SVC(kernel = modelsHyperParameters[j]["kernel"], 
                degree = modelsHyperParameters[j]["kernelDegree"], 
                gamma = modelsHyperParameters[j]["kernelGamma"], 
                C = modelsHyperParameters[j]["C"],
                probability=true);
                
                # Saving the model before training it
                models_dict["SVM $j"] = model_SVM;

                #Training the model
                # fit!(model_SVM, inputs_training, targets_training);
                
                # #Printing the accuracy
                # acc = score(model_SVM, inputs_test, targets_test);
                # println("SVM $j: $(acc*100) %");


            
            #Analogous code for the kNN model
            elseif modelsHyperParameters[j]["model"] == "kNN"

                model_kNN = KNeighborsClassifier(n_neighbors = modelsHyperParameters[j]["n_neighbors"]);
                
                models_dict["kNN $j"] = model_kNN;

                #Trainig the model
                # fit!(model_kNN, inputs_training, targets_training);
                

                # acc = score(model_kNN, inputs_test, targets_test);
                # println("kNN $j: $(acc*100) %");
                

            #Analogous code for the Decision Tree model
            elseif modelsHyperParameters[j]["model"] =="DecisionTree"
            
                #Defining the model
                model_DT = DecisionTreeClassifier(max_depth = modelsHyperParameters[j]["max_depth"], random_state = modelsHyperParameters[j]["random_state"]);
                
                models_dict["DecisionTree $j"] = model_DT;

                #Training the model
                # fit!(model_DT, inputs_training, targets_training);

                # acc = score(model_DT, inputs_test, targets_test);
                # println("Decision Tree $j: $(acc*100) %");
                

            #In case of a wrong input
            else 
                println("Introducción del modelo erronea");
            end
        end 



        models_dict["Ensemble (Stacking)"] = StackingClassifier(estimators = [(name,models_dict[name]) for name in keys(models_dict)], final_estimator=SVC(probability=true), n_jobs=1);

        ScikitLearn.fit!(models_dict["Ensemble (Stacking)"], inputs_training_pca, targets_training); 


        acc = ScikitLearn.score(models_dict["Ensemble (Stacking)"], inputs_test_pca, targets_test);
        ens_acc[i] = acc;
        outputs_test = ScikitLearn.predict(models_dict["Ensemble (Stacking)"], inputs_test_pca);


        size_file_path = joinpath(folder_path, "size$i.jld2");
        JLD2.save(size_file_path,"siz", siz); 

        model1_file_path = joinpath(folder_path, "targets_ensemblemodel$i.jld2");
        JLD2.save(model1_file_path,"targets_test",targets_test);

        
        model2_file_path = joinpath(folder_path, "outputs_ensemblemodel$i.jld2");
        JLD2.save(model2_file_path,"outputs_model",outputs_test);


        metrics = confusionMatrix(outputs_test, targets_test);
        ens_f1[i] = metrics[7];

        println("Ensemble (Stacking): $(acc*100) %")
        
    end


    #Printing all the results (the mean and the standard deviation of each metric for each model)
    println("__________________________________________________________________");
    println("");
    println("Results:");
    println("");
    println("Accuracy:")
    mean_ens_acc = mean(ens_acc); st_dev_ens_acc = std(ens_acc); println("The ensemble model (Stacking) have a mean accuracy of: $mean_ens_acc with a standard deviation of: $st_dev_ens_acc");

    println("");
    println("F-1: ");
    mean_ens_f1 = mean(ens_f1); st_dev_ens_f1 = std(ens_f1); println("The ensemble model (Stacking) have a mean f1 of: $mean_ens_f1 with a standard deviation of: $st_dev_ens_f1");


    return mean_ens_acc, st_dev_ens_acc, mean_ens_f1, st_dev_ens_f1
end


#Same functions as the ones in the Train_models.jl but calling the functions with the PCA technique

function Train_models_PCA(inputs::AbstractArray{<:Any,2}, targets::AbstractArray{<:Any,1}, index::Array{Int64,1})

    method = 1

    while method == 1 || method == 2 || method == 3 || method == 4 || method == 5
        println("Select: 1.- ANN, 2.- SVM, 3.- kNN, 4.- Decision Tree, 5.- Ensemble Model, Other.- Return")
        global method = parse(Int, readline())

        if method == 1
            println("ANN");
            TrainModelANN_PCA(inputs, targets,index);

        elseif method == 2
            println("SVM")
            TrainSVM_PCA(inputs, targets,index);

        elseif method == 3
            println("kNN");
            TrainkNN_PCA(inputs, targets,index);

        elseif method == 4
            println("DT");
            TrainDT_PCA(inputs,targets,index);

        elseif method == 5
            println("Ensemble");
            parameters = [Dict("model" => "kNN", "n_neighbors" => 1);
                            Dict("model" => "DecisionTree", "max_depth" => 30, "random_state" => 1);
                            Dict("model" => "DecisionTree", "max_depth" => 30, "random_state" => 1);];
            parameters = Vector{Dict}(parameters);
            estimators = [:kNN, :DecisionTree, :DecisionTree];
            trainClassEnsemble_PCA(estimators, parameters, (inputs, targets), index);
        else 
            return
        end
    end
end

function TrainModelANN_PCA(inputs::AbstractArray{<:Any,2}, targets::AbstractArray{<:Any,1}, index::Array{Int64,1})

    hidden_layers = 1;
    model_ann1 = 1;
    model_ann2 = 1;

    while hidden_layers == 1 || hidden_layers == 2
        println("Select: 1.- 1 Hidden Layer, 2.- 2 Hidden Layers, Other.- Return")
        global hidden_layers = parse(Int,readline())

        if hidden_layers == 1

            model_ann1 = 1;

            while model_ann1 == 1 || model_ann1 == 2 || model_ann1 == 3 || model_ann1 == 4 || model_ann1 == 5 || model_ann1 == 6 || model_ann1 == 7 || model_ann1 == 8 
                print("Select model: \n 1.- [15] softmax \n 2.- [15] tanh \n 3.- [15] sigmoid \n 4.- [15] relu \n 5.- [30] relu \n 6.- [120] relu \n 7.- [240] relu \n 8.- [480] relu \n 9.- Choose the hyperparameters \n Other.- Return \n")
                global model_ann1 = parse(Int, readline())

                if model_ann1 == 1
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [15], "transferFunctions" => fill(softmax, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index);
            
                elseif model_ann1 == 2
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [15], "transferFunctions" => fill(tanh, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann1 == 3
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [15], "transferFunctions" => fill(sigmoid, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann1 == 4
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [15], "transferFunctions" => fill(relu, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann1 == 5
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [30], "transferFunctions" => fill(relu, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann1 == 6
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [120], "transferFunctions" => fill(relu, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann1 == 7
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [240], "transferFunctions" => fill(relu, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann1 == 8
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [480], "transferFunctions" => fill(relu, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann1 == 9

                    println(" WARNING: SOME METHODS COULD HAVE EXCESIVE EXECUTION TIME ")
                    println("Enter the number of neurons you want in the first hidden layer: ")
                    num_neur1 = parse(Int, readline())

                    trans_fun = 1
                    while trans_fun == 1 || trans_fun == 2 || trans_fun == 3 || trans_fun == 4
                        println("Select the tranfer function: \n 1.- relu \n 2.- tanh \n 3.- sigmoid \n 4.- softmax.")
                        trans_fun = parse(Int, readline())
                    
                        if trans_fun == 1
                            println("Training the model...")
                            Random.seed!(123);
                            parameters_ann = Dict("topology" => [num_neur1], "transferFunctions" => fill(relu, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                            modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index) 

                        elseif trans_fun == 2
                            println("Training the model...")
                            Random.seed!(123);
                            parameters_ann = Dict("topology" => [num_neur1], "transferFunctions" => fill(tanh, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                            modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)     

                        elseif trans_fun == 3
                            println("Training the model...")
                            Random.seed!(123);
                            parameters_ann = Dict("topology" => [num_neur1], "transferFunctions" => fill(sigmoid, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                            modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)  

                        elseif trans_fun == 4
                            println("Training the model...")
                            Random.seed!(123);
                            parameters_ann = Dict("topology" => [num_neur1], "transferFunctions" => fill(softmax, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                            modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)   

                        end
                    end

                else
                    return 

                end 

            end 

        elseif hidden_layers == 2

            model_ann2 == 1

            while model_ann2 == 1 || model_ann2 == 2 || model_ann2 == 3 || model_ann2 == 4 || model_ann2 == 5 || model_ann2 == 6 || model_ann2 == 7 || model_ann2 == 8 
                print("Select model: \n 1.- [15,15] softmax \n 2.- [15,15] tanh \n 3.- [15,15] sigmoid \n 4.- [15,15] relu \n 5.- [120,120] relu  \n 6.- [240,240] relu \n 7.- [480,480] relu \n 8.- [240,240] sigmoid + relu \n 9.- [480,240] sigmoid + relu \n 10.- Choose the hyperparameters \n Other.- Return \n")
                global model_ann2 = parse(Int, readline())

                if model_ann2 == 1
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [15,15], "transferFunctions" => fill(softmax, 2), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index) 

                elseif model_ann2 == 2
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [15,15], "transferFunctions" => fill(tanh, 2), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index) 


                elseif model_ann2 == 3
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [15,15], "transferFunctions" => fill(sigmoid, 2), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann2 == 4
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [15,15], "transferFunctions" => fill(relu, 2), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann2 == 5
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [120,120], "transferFunctions" => fill(relu, 2), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann2 == 6
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [240,240], "transferFunctions" => fill(relu, 2), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann2 == 7
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [480,480], "transferFunctions" => fill(relu, 2), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann2 == 8
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [240,240], "transferFunctions" => [sigmoid, relu], "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann2 == 8
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [480,240], "transferFunctions" => [sigmoid, relu], "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann1 == 10
                    println(" WARNING: SOME METHODS COULD HAVE EXCESIVE EXECUTION TIME ")

                    println("Enter the number of neurons you want in the first hidden layer: ")
                    num_neur1 = parse(Int, readline())

                    println("Enter the number of neurons you want in the second hidden layer: ")
                    num_neur2 = parse(Int, readline())

                    trans_fun = 1
                    while trans_fun == 1 || trans_fun == 2 || trans_fun == 3 || trans_fun == 4
                        println("Select the tranfer function: \n 1.- relu \n 2.- tanh \n 3.- sigmoid \n 4.- softmax.")
                        trans_fun = parse(Int, readline())
                    
                        if trans_fun == 1
                            println("Training the model...")
                            Random.seed!(123);
                            parameters_ann = Dict("topology" => [num_neur1, num_neur2], "transferFunctions" => fill(relu, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                            modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)  

                        elseif trans_fun == 2
                            println("Training the model...")
                            Random.seed!(123);
                            parameters_ann = Dict("topology" => [num_neur1, num_neur2], "transferFunctions" => fill(tanh, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                            modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)  

                        elseif trans_fun == 3
                            println("Training the model...")
                            Random.seed!(123);
                            parameters_ann = Dict("topology" => [num_neur1, num_neur2], "transferFunctions" => fill(sigmoid, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                            modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index) 

                        elseif trans_fun == 4
                            println("Training the model...")
                            Random.seed!(123);
                            parameters_ann = Dict("topology" => [num_neur1], "transferFunctions" => fill(softmax, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                            modelCrossValidation_PCA(:ANN, parameters_ann, inputs, targets, index)     
                               
                        end
                    end

                else 
                    return
                end 

            end 
        else 
            return

        end 
    end
end

function TrainSVM_PCA(inputs::AbstractArray{<:Any,2}, targets::AbstractArray{<:Any,1}, index::Array{Int64,1})

    kernel_svm = 1;
    pol_deg = 1;
    rbf_hyp = 1;
    sigmoid_hyp = 1;

    while kernel_svm == 1 || kernel_svm == 2 || kernel_svm == 3 || kernel_svm == 4
        println("Select: 1.- Linear, 2.- Polinomial, 3.- Rbf, 4.- Sigmoid");
        println("WARNING: For some Approaches Linear and Polinomial kernel could take too long");
        global kernel_svm = parse(Int,readline())

        if kernel_svm == 1
            println("Linear");

            println("Select: \n 1.- C=1 \n 2.- Define your own C")
            svm_c = parse(Int,readline())

            if svm_c == 1
                parameters_svm = Dict("kernel" => "linear", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 1); 
                modelCrossValidation_PCA(:SVM, parameters_svm, inputs, targets, index);  

            elseif svm_c ==2
                println("Choose your c: ")
                local c = parse(Int, readline())

                parameters_svm = Dict("kernel" => "linear", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 1); 
                modelCrossValidation_PCA(:SVM, parameters_svm, inputs, targets, index);  

            end 

        elseif kernel_svm == 2
            println("Polinomial");

            global pol_deg = 1

            while pol_deg == 1 || pol_deg == 2 || pol_deg == 3
                println("Select: \n 1.- Polinomial Degree 2, C=1 \n 2.- Polinomial Degree 3, C=1 \n 3.- Polinomial Degree 5, C=1 \n 4.- Select your own Degree and C \n Other.- Return");
                global pol_deg = parse(Int,readline());
                
                if pol_deg == 1
                    println("Polinomial Degree 2 \n")
                    parameters_svm = Dict("kernel" => "poly", "kernelDegree" => 2, "kernelGamma" => 2, "C" => 1); 
                    modelCrossValidation_PCA(:SVM, parameters_svm, inputs, targets, index);

                elseif pol_deg == 2
                    println("Polinomial Degree 3\n")
                    parameters_svm = Dict("kernel" => "poly", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 1); 
                    modelCrossValidation_PCA(:SVM, parameters_svm, inputs, targets, index);

                elseif pol_deg == 3
                    println("Polinomial Degree 5\n")
                    parameters_svm = Dict("kernel" => "poly", "kernelDegree" => 5, "kernelGamma" => 2, "C" => 1); 
                    modelCrossValidation_PCA(:SVM, parameters_svm, inputs, targets, index);

                elseif pol_deg == 4
                    println("Choose the c");
                    local c = parse(Int,readline())

                    println("Choose the degree");
                    local deg = parse(Int, readline())

                    parameters_svm = Dict("kernel" => "poly", "kernelDegree" => deg, "kernelGamma" => 2, "C" => c); 
                    modelCrossValidation_PCA(:SVM, parameters_svm, inputs, targets, index);

                else 
                    return
                end 
            end

        elseif kernel_svm == 3
            println("Rbf");

            global rbf_hyp = 1

            while rbf_hyp == 1 || rbf_hyp == 2 || rbf_hyp == 3 || rbf_hyp == 4
                println("Select: \n 1.- Rbf c=1, gamma=3 \n 2.- Rbf c=1, gamma=20 \n 3.- Rbf c=500, gamma=10 \n 4.- Rbf c=1000, gamma=20 \n 5.- Choose your own hyperparameters \n Other.- Return");
                global rbf_hyp = parse(Int,readline())

                if rbf_hyp == 1
                    println("Rbf c=1, gamma=3\n")
                    parameters_svm = Dict("kernel" => "rbf", "kernelDegree" => 3, "kernelGamma" => 3, "C" => 1); 
                    modelCrossValidation_PCA(:SVM, parameters_svm, inputs, targets, index);

                elseif rbf_hyp == 2
                    println("Rbf c=1, gamma=20\n")
                    parameters_svm = Dict("kernel" => "rbf", "kernelDegree" => 3, "kernelGamma" => 20, "C" => 1); 
                    modelCrossValidation_PCA(:SVM, parameters_svm, inputs, targets, index);

                elseif rbf_hyp == 3
                    println("Rbf c=500, gamma=10\n")
                    parameters_svm = Dict("kernel" => "rbf", "kernelDegree" => 3, "kernelGamma" => 10, "C" => 500); 
                    modelCrossValidation_PCA(:SVM, parameters_svm, inputs, targets, index);

                elseif rbf_hyp == 4
                    println("Rbf c=1000, gamma=20\n")
                    parameters_svm = Dict("kernel" => "rbf", "kernelDegree" => 3, "kernelGamma" => 20, "C" => 1000); 
                    modelCrossValidation_PCA(:SVM, parameters_svm, inputs, targets, index);

                elseif rbf_hyp == 5
                    println("Choose the c");
                    local c = parse(Int,readline())

                    println("Choose the gamma");
                    local gamma = parse(Int, readline())

                    parameters_svm = Dict("kernel" => "rbf", "kernelDegree" => 3, "kernelGamma" => gamma, "C" => c); 
                    modelCrossValidation_PCA(:SVM, parameters_svm, inputs, targets, index);

                else 
                    return 
                end

            end
            
        elseif kernel_svm == 4
            println("Sigmoid");

            global sigmoid_hyp = 1

            while sigmoid_hyp == 1 || sigmoid_hyp == 2 || sigmoid_hyp == 3
                println("Select: \n 1.- Sigmoid c=1 \n 2.- Sigmoid c=500 \n 3.- Sigmoid c=1000 \n 4.- Choose your own hyperparameters \n Other.- Return")
                global sigmoid_hyp = parse(Int,readline())

                if sigmoid_hyp == 1
                    println("Sigmoid c=1\n")
                    parameters_svm = Dict("kernel" => "sigmoid", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 1); 
                    modelCrossValidation_PCA(:SVM, parameters_svm, inputs, targets, index)

                elseif sigmoid_hyp == 2
                    println("Sigmoid c=500\n")
                    parameters_svm = Dict("kernel" => "sigmoid", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 500); 
                    modelCrossValidation_PCA(:SVM, parameters_svm, inputs, targets, index)

                elseif sigmoid_hyp == 3
                    println("Sigmoid c=1000\n")
                    parameters_svm = Dict("kernel" => "sigmoid", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 1000); 
                    modelCrossValidation_PCA(:SVM, parameters_svm, inputs, targets, index)

                elseif sigmoid_hyp == 4
                    println("Choose the c");
                    local c = parse(Int,readline())

                    parameters_svm = Dict("kernel" => "sigmoid", "kernelDegree" => 3, "kernelGamma" => 2, "C" => c); 
                    modelCrossValidation_PCA(:SVM, parameters_svm, inputs, targets, index)

                else 
                    return 
                end 
            end 
        else 
            return
        end
    end 


end

function TrainkNN_PCA(inputs::AbstractArray{<:Any,2}, targets::AbstractArray{<:Any,1}, index::Array{Int64,1})

    model_knn = 1

    while model_knn == 1 || model_knn == 2 || model_knn == 3 || model_knn == 4 || model_knn == 5 || model_knn == 6
        println(" ");
        println("Select:\n 1.- kNN k=1 \n 2.- kNN k=2 \n 3.- kNN k=3 \n 4.- kNN k=5 \n 5.- kNN k=10 \n 6.- kNN k=15 \n 7.- Select your own hyperparameters \n Other.- Return")
        global model_knn = parse(Int,readline())
        
        if model_knn == 1 
            println("kNN k=1 \n")
            parameters_knn = Dict("n_neighbors" => 1);
            modelCrossValidation_PCA(:kNN, parameters_knn, inputs, targets, index); 

        elseif model_knn == 2
            println("kNN k=2 \n")
            parameters_knn = Dict("n_neighbors" => 2);
            modelCrossValidation_PCA(:kNN, parameters_knn, inputs, targets, index); 

        elseif model_knn == 3
            println("kNN k=3 \n")
            parameters_knn = Dict("n_neighbors" => 3);
            modelCrossValidation_PCA(:kNN, parameters_knn, inputs, targets, index); 

        elseif model_knn == 4
            println("kNN k=5 \n")
            parameters_knn = Dict("n_neighbors" => 5);
            modelCrossValidation_PCA(:kNN, parameters_knn, inputs, targets, index); 

        elseif model_knn == 5
            println("kNN k=10 \n")
            parameters_knn = Dict("n_neighbors" => 10);
            modelCrossValidation_PCA(:kNN, parameters_knn, inputs, targets, index); 

        elseif model_knn == 6
            println("kNN k=15 \n")
            parameters_knn = Dict("n_neighbors" => 15);
            modelCrossValidation_PCA(:kNN, parameters_knn, inputs, targets, index); 

        elseif model_knn == 7
            println("Choose your k: ") 
            local neigh = parse(Int, readline())

            parameters_knn = Dict("n_neighbors" => neigh);
            modelCrossValidation_PCA(:kNN, parameters_knn, inputs, targets, index);

        else 
            return 
        end 
    end 
end 

function TrainDT_PCA(inputs::AbstractArray{<:Any,2}, targets::AbstractArray{<:Any,1}, index::Array{Int64,1})

    model_dt = 1

    while model_dt == 1 || model_dt == 2 || model_dt == 3 || model_dt == 4 || model_dt == 5 || model_dt == 6
        println(" ");
        println("Select: \n 1.- DT depth=5 \n 2.- DT depth=20 \n 3.- DT depth=30 \n 4.- DT depth=40 \n 5.- DT depth=50 \n 6.- DT depth=100 \n 7.- Choose the hyperparameters  \n Other.- Return")
        global model_dt = parse(Int,readline())
        
        if model_dt == 1 
            println("DT depth=5 \n")
            parameters_dt = Dict("max_depth" => 5, "random_state" => 1);
            modelCrossValidation_PCA(:DecisionTree, parameters_dt, inputs, targets, index)

        elseif model_dt == 2
            println("DT depth=20 \n")
            parameters_dt = Dict("max_depth" => 20, "random_state" => 1);
            modelCrossValidation_PCA(:DecisionTree, parameters_dt, inputs, targets, index)

        elseif model_dt == 3
            println("DT depth=30 \n")
            parameters_dt = Dict("max_depth" => 30, "random_state" => 1);
            modelCrossValidation_PCA(:DecisionTree, parameters_dt, inputs, targets, index)

        elseif model_dt == 4
            println("DT depth=40 \n")
            parameters_dt = Dict("max_depth" => 40, "random_state" => 1);
            modelCrossValidation_PCA(:DecisionTree, parameters_dt, inputs, targets, index)

        elseif model_dt == 5
            println("DT depth=50 \n")
            parameters_dt = Dict("max_depth" => 50, "random_state" => 1);
            modelCrossValidation_PCA(:DecisionTree, parameters_dt, inputs, targets, index)

        elseif model_dt == 6
            println("DT depth=100 \n")
            parameters_dt = Dict("max_depth" => 100, "random_state" => 1);
            modelCrossValidation_PCA(:DecisionTree, parameters_dt, inputs, targets, index)

        elseif model_dt == 7
            println("Choose your depth: ") 
            local depth = parse(Int, readline())
            parameters_dt = Dict("max_depth" => depth, "random_state" => 1);
            modelCrossValidation_PCA(:DecisionTree, parameters_dt, inputs, targets, index)
            

        else 
            return 
        end 
    end 
end 