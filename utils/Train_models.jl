#________________________________________________________________________________________________________________#
#________________________________________________________________________________________________________________#
#                                   Functions for train the models                                               #
#________________________________________________________________________________________________________________#
#________________________________________________________________________________________________________________#



# General function for select the model which want to train and call the next functions
function Train_models(inputs::AbstractArray{<:Any,2}, targets::AbstractArray{<:Any,1}, index::Array{Int64,1})

    method = 1

    #Selecting the method
    while method == 1 || method == 2 || method == 3 || method == 4 || method == 5
        println("Select: 1.- ANN, 2.- SVM, 3.- kNN, 4.- Decision Tree, 5.- Ensemble Model, Other.- Return")
        global method = parse(Int, readline())

        #ANN and call the function for train ANNs
        if method == 1
            println("ANN");
            TrainModelANN(inputs, targets,index);

        #SVM and call the function for train the SVM 
        elseif method == 2
            println("SVM")
            TrainSVM(inputs, targets,index);

        #kNN and call the function for train the kNN
        elseif method == 3
            println("kNN");
            TrainkNN(inputs, targets,index);

        #Decision Tree and call the function for train the DT
        elseif method == 4
            println("DT");
            TrainDT(inputs,targets,index);

        #Ensemble model, train without call any function, much shorter
        elseif method == 5
            println("Ensemble");
            parameters = [Dict("model" => "kNN", "n_neighbors" => 1);
                            Dict("model" => "DecisionTree", "max_depth" => 30, "random_state" => 1);
                            Dict("model" => "DecisionTree", "max_depth" => 30, "random_state" => 1);];
            parameters = Vector{Dict}(parameters);
            estimators = [:kNN, :DecisionTree, :DecisionTree];
            trainClassEnsemble(estimators, parameters, (inputs, targets), index);
        else 
            return
        end
    end
end

# Function for train de ANN models 
function TrainModelANN(inputs::AbstractArray{<:Any,2}, targets::AbstractArray{<:Any,1}, index::Array{Int64,1})

    #Initialize some variables
    hidden_layers = 1;
    model_ann1 = 1;
    model_ann2 = 1;

    #Selecting one or two hidden layers
    while hidden_layers == 1 || hidden_layers == 2
        println("Select: 1.- 1 Hidden Layer, 2.- 2 Hidden Layers, Other.- Return")
        global hidden_layers = parse(Int,readline())

        #ONE HIDDEN LAYER
        if hidden_layers == 1

            model_ann1 = 1;

            #Selecting the model between the models used in the project or a new model with new parameters
            while model_ann1 == 1 || model_ann1 == 2 || model_ann1 == 3 || model_ann1 == 4 || model_ann1 == 5 || model_ann1 == 6 || model_ann1 == 7 || model_ann1 == 8 || model_ann1 == 9

                print("Select model: \n 1.- [15] softmax \n 2.- [15] tanh \n 3.- [15] sigmoid \n 4.- [15] relu \n 5.- [30] relu  \n 6.- [120] relu \n 7.- [240] relu \n 8.- [480] relu \n 9.- Select your hyperparameters \n Other.- Return \n")
                global model_ann1 = parse(Int, readline())


                #Selecting the model and calling the functions developed in the units for train it
                if model_ann1 == 1
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [15], "transferFunctions" => fill(softmax, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelos = modelCrossValidation(:ANN, parameters_ann, inputs, targets, index);
            
                elseif model_ann1 == 2
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [15], "transferFunctions" => fill(tanh, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann1 == 3
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [15], "transferFunctions" => fill(sigmoid, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann1 == 4
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [15], "transferFunctions" => fill(relu, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann1 == 5
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [30], "transferFunctions" => fill(relu, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann1 == 6
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [120], "transferFunctions" => fill(relu, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann1 == 7
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [240], "transferFunctions" => fill(relu, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann1 == 8
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [480], "transferFunctions" => fill(relu, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)

                #Defining the new model selecting the number of neurons and the transfer function
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
                            modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)        
                        elseif trans_fun == 2
                            println("Training the model...")
                            Random.seed!(123);
                            parameters_ann = Dict("topology" => [num_neur1], "transferFunctions" => fill(tanh, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                            modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)        
                        elseif trans_fun == 3
                            println("Training the model...")
                            Random.seed!(123);
                            parameters_ann = Dict("topology" => [num_neur1], "transferFunctions" => fill(sigmoid, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                            modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)        
                        elseif trans_fun == 4
                            println("Training the model...")
                            Random.seed!(123);
                            parameters_ann = Dict("topology" => [num_neur1], "transferFunctions" => fill(softmax, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                            modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)        
                        end
                    end

                else
                    return 

                end 

            end 

        #TWO HIDDEN LAYERS (Analogous to the method with one hidden layer)
        elseif hidden_layers == 2

            model_ann2 == 1

            while model_ann2 == 1 || model_ann2 == 2 || model_ann2 == 3 || model_ann2 == 4 || model_ann2 == 5 || model_ann2 == 6 || model_ann2 == 7 || model_ann2 == 8 || model_ann2 == 9 || model_ann2 == 10
                
                print("Select model: \n 1.- [15,15] softmax \n 2.- [15,15] tanh \n 3.- [15,15] sigmoid \n 4.- [15,15] relu \n 5.- [120,120] relu  
                        \n 6.- [240,240] relu \n 7.- [480,480] relu \n 8.- [240,240] sigmoid + relu \n 9.- [480,240] sigmoid + relu \n Other.- Return \n")
                global model_ann2 = parse(Int, readline())

                if model_ann2 == 1
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [15,15], "transferFunctions" => fill(softmax, 2), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation(:ANN, parameters_ann, inputs, targets, index) 

                elseif model_ann2 == 2
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [15,15], "transferFunctions" => fill(tanh, 2), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation(:ANN, parameters_ann, inputs, targets, index) 


                elseif model_ann2 == 3
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [15,15], "transferFunctions" => fill(sigmoid, 2), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann2 == 4
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [15,15], "transferFunctions" => fill(relu, 2), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann2 == 5
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [120,120], "transferFunctions" => fill(relu, 2), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann2 == 6
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [240,240], "transferFunctions" => fill(relu, 2), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann2 == 7
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [480,480], "transferFunctions" => fill(relu, 2), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann2 == 8
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [240,240], "transferFunctions" => [sigmoid, relu], "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)

                elseif model_ann2 == 9
                    Random.seed!(123);
                    parameters_ann = Dict("topology" => [480,240], "transferFunctions" => [sigmoid, relu], "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                    modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)

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
                            modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)        
                        elseif trans_fun == 2
                            println("Training the model...")
                            Random.seed!(123);
                            parameters_ann = Dict("topology" => [num_neur1, num_neur2], "transferFunctions" => fill(tanh, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                            modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)        
                        elseif trans_fun == 3
                            println("Training the model...")
                            Random.seed!(123);
                            parameters_ann = Dict("topology" => [num_neur1, num_neur2], "transferFunctions" => fill(sigmoid, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                            modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)        
                        elseif trans_fun == 4
                            println("Training the model...")
                            Random.seed!(123);
                            parameters_ann = Dict("topology" => [num_neur1], "transferFunctions" => fill(softmax, 1), "learningRate" => 0.01, "repetitionsTraining" => 1, "validationRatio" => 0.0, "maxEpochsVal" => 20);
                            modelCrossValidation(:ANN, parameters_ann, inputs, targets, index)        
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

#Function for train the SVM models
function TrainSVM(inputs::AbstractArray{<:Any,2}, targets::AbstractArray{<:Any,1}, index::Array{Int64,1})

    #Initialize some variables
    kernel_svm = 1;
    pol_deg = 1;
    rbf_hyp = 1;
    sigmoid_hyp = 1;

    #Selecting the kernel for the model
    while kernel_svm == 1 || kernel_svm == 2 || kernel_svm == 3 || kernel_svm == 4

        println("Select: \n 1.- Linear \n 2.- Polinomial \n 3.- Rbf \n 4.- Sigmoid \n  Other.- Return");
        println("WARNING: For some Approaches Linear and Polinomial kernel could take too long");
        global kernel_svm = parse(Int,readline())

        #Linear kernel
        if kernel_svm == 1

            println("Linear");
            println("Select: \n 1.- C=1 \n 2.- Define your own C")
            svm_c = parse(Int,readline())

            #Use the same model as in the project or define a new model with diferent parameters
            if svm_c == 1
                parameters_svm = Dict("kernel" => "linear", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 1); 
                modelCrossValidation(:SVM, parameters_svm, inputs, targets, index);    

            elseif svm_c ==2
                println("Choose your c: ")
                local c = parse(Int, readline())

                parameters_svm = Dict("kernel" => "linear", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 1); 
                modelCrossValidation(:SVM, parameters_svm, inputs, targets, index);   

            end 

        #Polinomial kernel 
        elseif kernel_svm == 2

            println("Polinomial");
            global pol_deg = 1

            #Selecting the models used in the project or a new one
            while pol_deg == 1 || pol_deg == 2 || pol_deg == 3
                println("Select: \n 1.- Polinomial Degree 2, C=1 \n 2.- Polinomial Degree 3, C=1 \n 3.- Polinomial Degree 5, C=1 \n 4.- Select your own Degree and C \n Other.- Return");
                global pol_deg = parse(Int,readline());
                
                if pol_deg == 1
                    println("Polinomial Degree 2 \n")
                    parameters_svm = Dict("kernel" => "poly", "kernelDegree" => 2, "kernelGamma" => 2, "C" => 1); 
                    modelCrossValidation(:SVM, parameters_svm, inputs, targets, index);

                elseif pol_deg == 2
                    println("Polinomial Degree 3\n")
                    parameters_svm = Dict("kernel" => "poly", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 1); 
                    modelCrossValidation(:SVM, parameters_svm, inputs, targets, index);

                elseif pol_deg == 3
                    println("Polinomial Degree 5\n")
                    parameters_svm = Dict("kernel" => "poly", "kernelDegree" => 5, "kernelGamma" => 2, "C" => 1); 
                    modelCrossValidation(:SVM, parameters_svm, inputs, targets, index);

                elseif pol_deg == 4
                    println("Choose the c");
                    local c = parse(Int,readline())

                    println("Choose the degree");
                    local deg = parse(Int, readline())

                    parameters_svm = Dict("kernel" => "poly", "kernelDegree" => deg, "kernelGamma" => 2, "C" => c); 
                    modelCrossValidation(:SVM, parameters_svm, inputs, targets, index);

                else 
                    return
                end 
            end

        #Rbf kernel
        elseif kernel_svm == 3

            println("Rbf");
            global rbf_hyp = 1

            #Selecting one of the models used in the project or defining a new one
            while rbf_hyp == 1 || rbf_hyp == 2 || rbf_hyp == 3 || rbf_hyp == 4
                println("Select: \n 1.- Rbf c=1, gamma=3 \n 2.- Rbf c=1, gamma=20 \n 3.- Rbf c=500, gamma=10 \n 4.- Rbf c=1000, gamma=20 \n 5.- Choose your own hyperparameters \n Other.- Return");
                global rbf_hyp = parse(Int,readline())

                if rbf_hyp == 1
                    println("Rbf c=1, gamma=3\n")
                    parameters_svm = Dict("kernel" => "rbf", "kernelDegree" => 3, "kernelGamma" => 3, "C" => 1); 
                    modelCrossValidation(:SVM, parameters_svm, inputs, targets, index);

                elseif rbf_hyp == 2
                    println("Rbf c=1, gamma=20\n")
                    parameters_svm = Dict("kernel" => "rbf", "kernelDegree" => 3, "kernelGamma" => 20, "C" => 1); 
                    modelCrossValidation(:SVM, parameters_svm, inputs, targets, index);

                elseif rbf_hyp == 3
                    println("Rbf c=500, gamma=10\n")
                    parameters_svm = Dict("kernel" => "rbf", "kernelDegree" => 3, "kernelGamma" => 10, "C" => 500); 
                    modelCrossValidation(:SVM, parameters_svm, inputs, targets, index);

                elseif rbf_hyp == 4
                    println("Rbf c=1000, gamma=20\n")
                    parameters_svm = Dict("kernel" => "rbf", "kernelDegree" => 3, "kernelGamma" => 20, "C" => 1000); 
                    modelCrossValidation(:SVM, parameters_svm, inputs, targets, index);

                elseif rbf_hyp == 5
                    println("Choose the c");
                    local c = parse(Int,readline())

                    println("Choose the gamma");
                    local gamma = parse(Int, readline())

                    parameters_svm = Dict("kernel" => "rbf", "kernelDegree" => 3, "kernelGamma" => gamma, "C" => c); 
                    modelCrossValidation(:SVM, parameters_svm, inputs, targets, index);

                else 
                    return 
                end

            end
            
        #Sigmoid kernel
        elseif kernel_svm == 4

            println("Sigmoid");
            global sigmoid_hyp = 1

            #Selecting if uses a model from the project or defining a new one
            while sigmoid_hyp == 1 || sigmoid_hyp == 2 || sigmoid_hyp == 3
                println("Select: \n 1.- Sigmoid c=1 \n 2.- Sigmoid c=500 \n 3.- Sigmoid c=1000 \n 4.- Choose your own hyperparameters \n Other.- Return")
                global sigmoid_hyp = parse(Int,readline())

                if sigmoid_hyp == 1
                    println("Sigmoid c=1\n")
                    parameters_svm = Dict("kernel" => "sigmoid", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 1); 
                    modelCrossValidation(:SVM, parameters_svm, inputs, targets, index)

                elseif sigmoid_hyp == 2
                    println("Sigmoid c=500\n")
                    parameters_svm = Dict("kernel" => "sigmoid", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 500); 
                    modelCrossValidation(:SVM, parameters_svm, inputs, targets, index)

                elseif sigmoid_hyp == 3
                    println("Sigmoid c=1000\n")
                    parameters_svm = Dict("kernel" => "sigmoid", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 1000); 
                    modelCrossValidation(:SVM, parameters_svm, inputs, targets, index)

                elseif sigmoid_hyp == 4
                    println("Choose the c");
                    local c = parse(Int,readline())

                    parameters_svm = Dict("kernel" => "sigmoid", "kernelDegree" => 3, "kernelGamma" => 2, "C" => c); 
                    modelCrossValidation(:SVM, parameters_svm, inputs, targets, index)

                else 
                    return 
                end 
            end 
        else 
            return
        end
    end 


end

#Function for train kNN models
function TrainkNN(inputs::AbstractArray{<:Any,2}, targets::AbstractArray{<:Any,1}, index::Array{Int64,1})

    model_knn = 1

    #Selecting between a model used in the project or a new model 
    while model_knn == 1 || model_knn == 2 || model_knn == 3 || model_knn == 4 || model_knn == 5 || model_knn == 6

        println(" ");
        println("Select:\n 1.- kNN k=1 \n 2.- kNN k=2 \n 3.- kNN k=3 \n 4.- kNN k=5 \n 5.- kNN k=10 \n 6.- kNN k=15 \n 7.- Select your own hyperparameters \n Other.- Return")
        global model_knn = parse(Int,readline())
        
        if model_knn == 1 
            println("kNN k=1 \n")
            parameters_knn = Dict("n_neighbors" => 1);
            modelCrossValidation(:kNN, parameters_knn, inputs, targets, index);  

        elseif model_knn == 2
            println("kNN k=2 \n")
            parameters_knn = Dict("n_neighbors" => 2);
            modelCrossValidation(:kNN, parameters_knn, inputs, targets, index); 

        elseif model_knn == 3
            println("kNN k=3 \n")
            parameters_knn = Dict("n_neighbors" => 3);
            modelCrossValidation(:kNN, parameters_knn, inputs, targets, index); 

        elseif model_knn == 4
            println("kNN k=5 \n")
            parameters_knn = Dict("n_neighbors" => 5);
            modelCrossValidation(:kNN, parameters_knn, inputs, targets, index);

        elseif model_knn == 5
            println("kNN k=10 \n")
            parameters_knn = Dict("n_neighbors" => 10);
            modelCrossValidation(:kNN, parameters_knn, inputs, targets, index); 

        elseif model_knn == 6
            println("kNN k=15 \n")
            parameters_knn = Dict("n_neighbors" => 15);
            modelCrossValidation(:kNN, parameters_knn, inputs, targets, index);

        #Defining a new k for the model
        elseif model_knn == 7
            println("Choose your k: ") 
            local neigh = parse(Int, readline())

            parameters_knn = Dict("n_neighbors" => neigh);
            modelCrossValidation(:kNN, parameters_knn, inputs, targets, index);

        else 
            return 
        end 
    end 
end

#Function for train Decision Tree models
function TrainDT(inputs::AbstractArray{<:Any,2}, targets::AbstractArray{<:Any,1}, index::Array{Int64,1})

    model_dt = 1

    #Selecting between the models used in the project or defining a new model
    while model_dt == 1 || model_dt == 2 || model_dt == 3 || model_dt == 4 || model_dt == 5 || model_dt == 6

        println(" ");
        println("Select: \n 1.- DT depth=5 \n 2.- DT depth=20 \n 3.- DT depth=30 \n 4.- DT depth=40 \n 5.- DT depth=50 \n 6.- DT depth=100 \n 7.- Choose the hyperparameters  \n Other.- Return")
        global model_dt = parse(Int,readline())

        if model_dt == 1 
            println("DT depth=5 \n")
            parameters_dt = Dict("max_depth" => 5, "random_state" => 1);
            modelCrossValidation(:DecisionTree, parameters_dt, inputs, targets, index)

        elseif model_dt == 2
            println("DT depth=20 \n")
            parameters_dt = Dict("max_depth" => 20, "random_state" => 1);
            modelCrossValidation(:DecisionTree, parameters_dt, inputs, targets, index)

        elseif model_dt == 3
            println("DT depth=30 \n")
            parameters_dt = Dict("max_depth" => 30, "random_state" => 1);
            modelCrossValidation(:DecisionTree, parameters_dt, inputs, targets, index)

        elseif model_dt == 4
            println("DT depth=40 \n")
            parameters_dt = Dict("max_depth" => 40, "random_state" => 1);
            modelCrossValidation(:DecisionTree, parameters_dt, inputs, targets, index)

        elseif model_dt == 5
            println("DT depth=50 \n")
            parameters_dt = Dict("max_depth" => 50, "random_state" => 1);
            modelCrossValidation(:DecisionTree, parameters_dt, inputs, targets, index)

        elseif model_dt == 6
            println("DT depth=100 \n")
            parameters_dt = Dict("max_depth" => 100, "random_state" => 1);
            modelCrossValidation(:DecisionTree, parameters_dt, inputs, targets, index)

        #Defining a depth for a new model 
        elseif model_dt == 7
            println("Choose your depth: ") 
            local depth = parse(Int, readline())

            parameters_dt = Dict("max_depth" => depth, "random_state" => 1);
            modelCrossValidation(:DecisionTree, parameters_dt, inputs, targets, index)
            
        else 
            return 
        end 
    end 
end 