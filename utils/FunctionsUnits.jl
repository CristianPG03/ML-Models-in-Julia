#________________________________________________________________________________________________________________#
#________________________________________________________________________________________________________________#
#                                     Functions for developed in the Units                                       #
#________________________________________________________________________________________________________________#
#________________________________________________________________________________________________________________#







#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#                           UNIT 2                                         #
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#


function oneHotEncoding(feature::AbstractArray{<:Any,1},      
        classes::AbstractArray{<:Any,1})
    # First we are going to set a line as defensive to check values
    @assert(all([in(value, classes) for value in feature]));

    # Second defensive statement, check the number of classes
    numClasses = length(classes);
    # @assert(numClasses>1)

    if (numClasses==2)
        # Case with only two classes
        oneHot = reshape(feature.==classes[1], :, 1);
    else
        #Case with more than two clases
        oneHot =  BitArray{2}(undef, length(feature), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    end;
    return oneHot;
end;


oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end;


function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return mean(dataset, dims=1), std(dataset, dims=1)
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    # eliminate any atribute that do not add information
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset , calculateMinMaxNormalizationParameters(dataset));
end;

function normalizeMinMax( dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
normalizeMinMax!(copy(dataset), normalizationParameters);
end;

function normalizeMinMax( dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset));
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    avgValues = normalizationParameters[1];
    stdValues = normalizationParameters[2];
    dataset .-= avgValues;
    dataset ./= stdValues;
    dataset[:, vec(stdValues.==0)] .= 0;
    return dataset; 
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizeZeroMean!(dataset , calculateZeroMeanNormalizationParameters(dataset));   
end;

function normalizeZeroMean( dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    normalizeZeroMean!(copy(dataset), normalizationParameters);
end;

function normalizeZeroMean( dataset::AbstractArray{<:Real,2}) 
    normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset));
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; 
    threshold::Real=0.5) 
    numOutputs = size(outputs, 2);
    @assert(numOutputs!=2)
    if numOutputs==1
    return outputs.>=threshold;
    else
    (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
    # Set up then boolean matrix to everything false while max values aretrue.
    outputs = falses(size(outputs));
    outputs[indicesMaxEachInstance] .= true;
    # Defensive check if all patterns are in a single class
    @assert(all(sum(outputs, dims=2).==1));
    return outputs;
    end;
end;

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    return mean(outputs.==targets);
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};      
    threshold::Real=0.5)
    accuracy(outputs.>=threshold, targets);
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
    threshold::Real=0.5)
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
    return accuracy(outputs[:,1], targets[:,1]);
    else
    return accuracy(classifyOutputs(outputs; threshold=threshold), targets);
    end;
end;

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
    ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:length(topology)
    numNeurons = topology[numHiddenLayer];
    ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
    numInputsLayer = numNeurons;
    end;
    if (numOutputs == 1)
    ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
    ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
    ann = Chain(ann..., softmax);
    end;
    return ann;
end;  

function trainClassANN(topology::AbstractArray{<:Int,1},      
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) 

    (inputs, targets) = dataset;

    @assert(size(inputs,1)==size(targets,1));

    ann = buildClassANN(size(inputs,2), topology, size(targets,2));

    loss(model,x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);

    trainingLosses = Float32[];

    numEpoch = 0;
    trainingLoss = loss(ann, inputs', targets');
    push!(trainingLosses, trainingLoss);

    opt_state = Flux.setup(Adam(learningRate), ann);

    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)

    Flux.train!(loss, ann, [(inputs', targets')], opt_state);

    numEpoch += 1;

    trainingLoss = loss(ann, inputs', targets');

    push!(trainingLosses, trainingLoss);

    end;

    return (ann, trainingLosses);
end;                                        

function trainClassANN(topology::AbstractArray{<:Int,1},      
    (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};      
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),      
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    trainClassANN(topology, (inputs, reshape(targets, length(targets), 1)); 
    maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate);
end;

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#                           UNIT 3                                         #
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#


function holdOut(N::Int, P::Real)
    #TODO
    @assert(0<=P<1);
    
    indexes = randperm(N);
    perc = Int(ceil(N*P));
    test = indexes[1:perc];
    train = indexes[(perc+1):N];

    @assert((length(train) + length(test)) == N)
    
    return(train,test);
end

function holdOut(N::Int, Pval::Real, Ptest::Real)
    #TODO
    @assert((Pval + Ptest)<1);

    P_val_test = Pval + Ptest;

    N_val_test = Int(ceil(N*P_val_test));

    (train, val_test) = holdOut(N, P_val_test);
    (ind_val, ind_test) = holdOut(N_val_test, Pval/(1 - Ptest));

    val = val_test[ind_val];
    test = val_test[ind_test];

    @assert((length(val) + length(train) + length(test)) == N);
    
    return (train, val, test)
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
            (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
            (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20, showText::Bool=false)

    (train_inputs, train_targets) = trainingDataset;
    (val_inputs, val_targets) = validationDataset;
    (test_inputs, test_targets) = testDataset;


    @assert(size(train_inputs,1)==size(train_targets,1));
    @assert(size(val_inputs,1)==size(val_targets,1));
    @assert(size(test_inputs,1)==size(test_targets,1));

    ann = buildClassANN(size(train_inputs,2), topology, size(train_targets,2));
    annMinError = undef
    loss(model,x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);

    trainingLosses = Float32[];
    valLosses = Float32[];
    testLosses = Float32[];

    numEpoch = 0;
    numEpochVal = 0;
    minValLoss = 100;
    trainingLoss = loss(ann, train_inputs', train_targets');
    push!(trainingLosses, trainingLoss);

    valLoss = loss(ann, val_inputs', val_targets');
    push!(valLosses, valLoss);

    testLoss = loss(ann, test_inputs', test_targets');
    push!(testLosses, testLoss);

    println("Epoch ", numEpoch, ": train loss : ", trainingLoss, ": validation loss : ", valLoss, ": test loss :", testLoss);

    opt_state = Flux.setup(Adam(learningRate), ann);


    if isempty(validationDataset)

    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)

        Flux.train!(loss, ann, [(train_inputs', train_targets')], opt_state);

        numEpoch += 1;
        trainingLoss = loss(ann, train_inputs', train_targets');
        push!(trainingLosses, trainingLoss);
        testLoss = loss(ann, test_inputs', test_targets');
        push!(testLosses, testLoss);
            
    end;

    println("Epoch ", numEpoch, ": train loss: ", trainingLoss, ": validation loss :", valLoss, ": test loss :", testLoss);


    return (ann, trainingLosses, test,Losses)

    else

    while (numEpoch<maxEpochs) && (trainingLoss>minLoss) && (numEpochVal<maxEpochsVal)

        Flux.train!(loss, ann, [(train_inputs', train_targets')], opt_state);

        numEpoch += 1;

        numEpochVal+=1;
        
        trainingLoss = loss(ann, train_inputs', train_targets');
        push!(trainingLosses, trainingLoss);
        
        valLoss = loss(ann, val_inputs', val_targets');
        push!(valLosses, valLoss);

        testLoss = loss(ann, test_inputs', test_targets');
        push!(testLosses, testLoss);
        

        if valLoss < minValLoss

            minValLoss = valLoss;
            
            numEpochVal = 0;
            
            annMinError = deepcopy(ann);
        end;
            
    end;

    println("Epoch ", numEpoch, ": train loss: ", trainingLoss, ": validation loss :", valLoss, ": test loss :", testLoss);


    return (annMinError, trainingLosses, valLosses, testLosses)
        
    end;

end;                                        


function trainClassANN(topology::AbstractArray{<:Int,1},
    (train_inputs, train_targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
    (val_inputs, val_targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    (test_inputs, test_targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0));
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20, showText::Bool=false)

    trainClassANN(topology, (train_inputs, reshape(train_targets, length(train_targets), 1)), (val_inputs, reshape(val_targets, length(val_targets),1)),
        (test_inputs, reshape(test_targets, length(test_targets), 1)); transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss= minLoss,
        learningRate=learningRate, maxEpochsVal=maxEpochsVal, showText=showText)
end


#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#                           UNIT 4                                         #
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#



function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #TODO
    TP_vec = (outputs .& targets); #si output es true y targets es true guarda un 1
    TP = sum(TP_vec);
    TN_vec = (.!outputs .& .!targets)
    TN = sum(TN_vec)
    FN = sum(.!outputs) - TN
    FP = sum(outputs) - TP
    
    if (TN + TP + FN + FP) == 0
        acc = 0
    else
        acc = (TN + TP) / (TN + TP + FN + FP)
    end
    if (TN + TP + FN + FP) == 0
        er_rate = 0
    else
        er_rate = (FP + FN) / (TN + TP + FN + FP)
    end
    if  (FN + TP) == 0
        sen = 0
    else 
        sen = (TP) / (FN + TP)
    end
    if (FP + TN) == 0
        spe = 0
    else
        spe = (TN) / (FP + TN)
    end
    if (TP + FP) == 0
        prec = 0
    else
        prec = (TP) / (TP + FP)
    end
    if (TN + FN) == 0
        neg_pred = 0
    else 
        neg_pred = (TN) / (TN + FN)
    end
    if (prec + sen) == 0
        f_score = 0
    else
        f_score =  (2 * (prec * sen))/ (prec + sen)
    end
    mat_con = [[TN, FP] [FN, TP]]
    if acc == NaN
        acc = 0        
    end
    if er_rate == NaN
        er_rate = 0        
    end
    if sen == NaN
        sen = 0        
    end
    if spe == NaN
        spe = 0        
    end
    if prec == NaN
        prec = 0        
    end
    if neg_pred == NaN
        neg_pred = 0        
    end
    if f_score == NaN
        f_score = 0        
    end
    return acc,er_rate,sen,spe,prec,neg_pred,f_score, mat_con
end

function confusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #TODO
    outputs_b = (outputs.>=threshold)
    acc, er_rate, sen, spe, prec, neg_pred, f_score, mat_con = confusionMatrix(outputs_b,targets)
    return acc,er_rate,sen,spe,prec,neg_pred,f_score, mat_con
end

function printConfusionMatrix(outputs::AbstractArray{Bool,1},targets::AbstractArray{Bool,1})
    acc, er_rate, sen, spe, prec, neg_pred, f_score, mat_con = confusionMatrix(outputs, targets)
    print("Confusion Matrix: ")
    display(mat_con)
    print("1. Accuracy: ", acc, "\n")
    print("2. Error Rate: ", er_rate, "\n")
    print("3. Sensitivity / Recall: ", sen, "\n")
    print("4. Specificity: ", spe, "\n")
    print("5. Precision / Positive predictive value: ", prec, "\n")
    print("6. Negative predictive value: ", neg_pred, "\n")
    print("7. F-score: ", f_score, "\n")
    return
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs_b = (outputs.>=threshold)
    printConfusionMatrix(outputs_b,target)
    return
end

function oneVSall(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2})
    numClasses = size(targets,2);
    numInstances = size(targets,1);
    outputs = Array{Float32,2}(undef, numInstances, numClasses);
    for numClass in 1:numClasses
        outputs[:,numClass] = fit(inputs, targets[:,numClass]);
    end
    outputs = (softmax(outputs')');
    vmax = maximum(outputs, dims=2);
    outputs_bol = (outputs .== vmax)
    return outputs_bol
end

function fit(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,1}, 
        numEpoch::Int=0, maxEpochs::Int=500, minLoss::Real=0.0, learningRate::Real=0.01)
    ann = Chain(
    Dense(4,1,sigmoid))
    loss(m, inputs, targets) = Losses.binarycrossentropy(m(inputs), targets)
    learningRate = 0.01; #parametro entre (0.01,0.1)
    opt_state = Flux.setup(Adam(learningRate), ann)
    while (numEpoch<maxEpochs)
        Flux.train!(loss, ann, [(inputs', targets')], opt_state)
        numEpoch += 1;  
    end;
    outputs = ann(inputs')
    return outputs
end

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)

    numClasses = size(outputs, 2);
    numInstances = size(outputs, 1);

    sens = zeros(numClasses);
    spe = zeros(numClasses);
    PPV = zeros(numClasses);
    NPV = zeros(numClasses);
    F1 = zeros(numClasses);
    mat_con = zeros(numClasses,numClasses);
    
    for class in 1:numClasses
        confusionMatrix_class = confusionMatrix(outputs[:,class], targets[:,class])
        sens[class] = confusionMatrix_class[3]
        spe[class] = confusionMatrix_class[4]
        PPV[class] = confusionMatrix_class[5]
        NPV[class] = confusionMatrix_class[6]
        F1[class] = confusionMatrix_class[7]
    end 
    for i in 1:numClasses
        for j in 1:numClasses
            mat_con[i,j] = sum(outputs[:,i].&targets[:,j])
        end
    end
    sens_m = mean(sens);
    spe_m = mean(spe);
    PPV_m = mean(PPV);
    NPV_m = mean(NPV);
    F1_m = mean(F1);


    sens_w = 0;
    spe_w = 0;
    PPV_w = 0;
    NPV_w = 0;
    F1_w = 0;
    for i in 1:numClasses
        if sens[i] == NaN
            sens_w = sens_w;
        else
            sens_w = sens_w + (sens[i]*(sum(targets[:,i])));
        end
        if spe[i] == NaN
            spe_w = spe_w;
        else 
            spe_w = spe_w + (spe[i]*(sum(targets[:,i])));
        end
        if PPV[i] == NaN
            PPV_w = PPV_w;
        else
            PPV_w = PPV_w + (PPV[i]*(sum(targets[:,i])));
        end 
        if NPV[i] == NaN
            NPV_w = NPV_w;
        else
            NPV_w = NPV_w + (NPV[i]*(sum(targets[:,i])));
        end 
        if F1[i] == NaN
            F1_w = F1_w
        else 
            F1_w = F1_w + (F1[i]*(sum(targets[:,i])));
        end
    end     
    sens_w = sens_w / numInstances;
    spe_w = spe_w / numInstances;
    PPV_w = PPV_w / numInstances;
    NPV_w = NPV_w / numInstances;
    F1_w = F1_w / numInstances;

    acc = 0;
    for i in 1:numClasses
        ind = (targets[:,i] .== 1)
        acc = acc + sum(outputs[ind,i] .== targets[ind,i])
    end
    acc = acc/numInstances;
    err = 1 - acc;
    
    return mat_con, sens_w, spe_w, PPV_w, NPV_w, F1_w, sens_m, spe_m, PPV_m, NPV_m, F1_m, acc, err 
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # @assert(all([in(outputs, unique(targets)) for outputs in outputs]))
    outputs = vec(outputs);
    targets = vec(targets);
    outputs_c = outputs;
    targets_c = targets;
    numInstances = size(outputs,1);
    classes = vcat(outputs_c, targets_c);
    classes =  Set(classes);
    classes = collect(classes);
    numClasses = length(classes);
    outputs_bool =  BitArray{2}(undef, numInstances, numClasses);
    targets_bool =  BitArray{2}(undef, numInstances, numClasses);
    for numClass in 1:numClasses
        outputs_bool[:,numClass] .= (outputs.==classes[numClass]);
        targets_bool[:,numClass] .= (targets.==classes[numClass]);
    end;
    mat_con, sens_w, spe_w, PPV_w, NPV_w, F1_w, sens_m, spe_m, PPV_m, NPV_m, F1_m, acc, err = confusionMatrix(outputs_bool, targets_bool);
    return mat_con, sens_w, spe_w, PPV_w, NPV_w, F1_w, sens_m, spe_m, PPV_m, NPV_m, F1_m, acc, err 
end 

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    outputs = classifyOutputs(outputs);
    mat_con, sens_w, spe_w, PPV_w, NPV_w, F1_w, sens_m, spe_m, PPV_m, NPV_m, F1_m, acc, err = confusionMatrix(outputs, targets);
    return mat_con, sens_w, spe_w, PPV_w, NPV_w, F1_w, sens_m, spe_m, PPV_m, NPV_m, F1_m, acc, err 
end


#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#                           UNIT 5                                         #
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#



function crossvalidation(N::Int64, k::Int64)
    vec = Vector(1:k);
    dim = convert(Int, ceil(N/k));
    N_vect = repeat(vec, dim);
    vect = Vector(undef, N);
    for i in 1:N
        vect[i] = N_vect[i];
    end
    vect_shuff = shuffle!(vect)
    return vect_shuff
end


function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    nClasses = size(targets, 2);
    n = size(targets,1);
    index = Vector{Int}(undef, n);
    for class in 1:nClasses
        index[(targets[:,class] .== 1)] = crossvalidation(sum(targets[:,class]),k);
    end
    return index
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    targets_bool = oneHotEncoding(targets); 
    index = crossvalidation(targets_bool, k);
    return index
end 

function trainClassANN(topology::AbstractArray{<:Int,1}, 
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, 
    kFoldIndices::Array{Int64,1}; 
    transferFunctions::AbstractArray{<:Function,1}=fill(sigmoid, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, repetitionsTraining::Int=1, 
    validationRatio::Real=0.0, maxEpochsVal::Int=20)

    (inputs, targets) = trainingDataset;
    @assert(size(inputs,1)==size(targets,1));
    k = maximum(kFoldIndices);
    folder_path = "utils/ModelsTrained/modelann $topology, $transferFunctions";

    if !isdir(folder_path)
        mkpath(folder_path)
    end

    f_score = Vector(undef,k);
    acc = Vector(undef,k);


    for i in 1:k

        @assert((sum(kFoldIndices .!= i)) > 1); 

        nEl = sum(kFoldIndices .== i);
        n = size(inputs,1);

        inputs_tv = inputs[(kFoldIndices.!=i),:];
        targets_tv = targets[(kFoldIndices.!=i),:];
        train_ind, val_ind = holdOut( (n - nEl), 0.2)

        inputs_training = inputs_tv[train_ind , :];
        targets_training = targets_tv[train_ind , :];

        inputs_validation = inputs_tv[val_ind , :];
        targets_validation = targets_tv[val_ind ,:];
        
        inputs_test = inputs[(kFoldIndices.==i),:];
        targets_test = targets[(kFoldIndices.==i),:];

        
        println(" ");
        println("------------------------------------------------------------------------------------------------------------------");
        println("-------------------------------------------------- FOLD: ", i, " -------------------------------------------------------");
        println("------------------------------------------------------------------------------------------------------------------");

        (ann, annTrainLosses, annValLosses, annTestLosses) = trainClassANN(topology, (inputs_training, targets_training), 
        (inputs_validation, targets_validation),(inputs_test, targets_test) , transferFunctions, maxEpochs, minLoss, learningRate, maxEpochsVal);

        outputs_model = ann(inputs_test');
        outputs_model = outputs_model'

        model1_file_path = joinpath(folder_path, "targets_annmodel$i.jld2");
        JLD2.save(model1_file_path,"targets_test",targets_test);

        
        model2_file_path = joinpath(folder_path, "outputs_annmodel$i.jld2");
        JLD2.save(model2_file_path,"outputs_model",outputs_model);


        metric_ann = confusionMatrix(outputs_model, targets_test);
        
        println(" ");
        println("Metrics: ");
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


function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
    kFoldIndices::	Array{Int64,1};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,repetitionsTraining::Int=1, 
    validationRatio::Real=0.0, maxEpochsVal::Int=20)


    (inputs, targets) = trainingDataset
    targets = reshape(targets, length(train_targets), 1)

    return trainClassANN(topology, (inputs, targets), kFoldIndices=kFoldIndices; transferFunctions=transferFunctions, 
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, repetitionsTraining=repetitionsTraining, 
        validationRatio=validationRatio, maxEpochsVal=maxEpochsVal)
end

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#                           UNIT 6                                         #
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#


function modelCrossValidation(modelType::Symbol,
        modelHyperparameters::Dict,
        inputs::AbstractArray{<:Real,2},
        targets::AbstractArray{<:Any,1},
        crossValidationIndices::Array{Int64,1})
    
    k = maximum(crossValidationIndices);
    f_score = Vector(undef, k);
    acc = Vector(undef, k);
    
    if (modelType == :ANN)

        targets = oneHotEncoding(targets);

        model = trainClassANN(modelHyperparameters["topology"], (inputs, targets), crossValidationIndices; 
            transferFunctions = modelHyperparameters["transferFunctions"], maxEpochs = 1000, minLoss = 0.0,
            learningRate = modelHyperparameters["learningRate"], repetitionsTraining = modelHyperparameters["repetitionsTraining"],
            validationRatio = modelHyperparameters["validationRatio"], maxEpochsVal = modelHyperparameters["maxEpochsVal"])

        
    elseif (modelType == :SVM)

        for i in 1:k

            model = SVC(kernel = modelHyperparameters["kernel"], 
            degree = modelHyperparameters["kernelDegree"], 
            gamma = modelHyperparameters["kernelGamma"], 
            C = modelHyperparameters["C"]);

            kernel = modelHyperparameters["kernel"];
            c = modelHyperparameters["C"];
            gamma = modelHyperparameters["kernelGamma"];
            degree = modelHyperparameters["kernelDegree"];

            folder_path = "utils/ModelsTrained/modelsmodelsvm $kernel, $c, $gamma, $degree";

            if !isdir(folder_path)
                mkpath(folder_path)
            end

            @assert((sum(crossValidationIndices .!= i)) > 1); 
        
            nEl = sum(crossValidationIndices .== i);
            n = size(inputs,1);
        
            inputs_training = inputs[(crossValidationIndices.!=i),:];
            targets_training = reshape(targets[(crossValidationIndices.!=i),:], sum(crossValidationIndices.!=i));
            inputs_test = inputs[(crossValidationIndices.==i),:];
            targets_test = reshape(targets[(crossValidationIndices.==i),:], sum(crossValidationIndices.==i));
            
            println(" ");
            println("------------------------------------------------------------------------------------------------------------------");
            println("-------------------------------------------------- FOLD: ", i, " -------------------------------------------------------");
            println("------------------------------------------------------------------------------------------------------------------");

            ScikitLearn.fit!(model, inputs_training, targets_training);

            outputs_model = ScikitLearn.predict(model, inputs_test);


            model1_file_path = joinpath(folder_path, "targets_svmmodel$i.jld2");
            JLD2.save(model1_file_path,"targets_test",targets_test);

            
            model2_file_path = joinpath(folder_path, "outputs_svmmodel$i.jld2");
            JLD2.save(model2_file_path,"outputs_model",outputs_model);

            metric_svm = confusionMatrix(outputs_model, targets_test);
            
            println(" ");
            println("Metrics: ");
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



    elseif (modelType == :kNN)

        for i in 1:k

            model = KNeighborsClassifier(n_neighbors = modelHyperparameters["n_neighbors"]);

            k = modelHyperparameters["n_neighbors"]
            
            folder_path = "utils/ModelsTrained/modelknn $k";

            if !isdir(folder_path)
                mkpath(folder_path)
            end

            @assert((sum(crossValidationIndices .!= i)) > 1); 
        
            nEl = sum(crossValidationIndices .== i);
            n = size(inputs,1);
        
            inputs_training = inputs[(crossValidationIndices.!=i),:];
            targets_training = reshape(targets[(crossValidationIndices.!=i),:], sum(crossValidationIndices.!=i));
            inputs_test = inputs[(crossValidationIndices.==i),:];
            targets_test = reshape(targets[(crossValidationIndices.==i),:], sum(crossValidationIndices.==i));


            println(" ");
            println("------------------------------------------------------------------------------------------------------------------");
            println("-------------------------------------------------- FOLD: ", i, " -------------------------------------------------------");
            println("------------------------------------------------------------------------------------------------------------------");

            ScikitLearn.fit!(model, inputs_training, targets_training);
            
            outputs_model = ScikitLearn.predict(model, inputs_test);

            model1_file_path = joinpath(folder_path, "targets_knnmodel$i.jld2");
            JLD2.save(model1_file_path,"targets_test",targets_test);

            model2_file_path = joinpath(folder_path, "outputs_knnmodel$i.jld2");
            JLD2.save(model2_file_path,"outputs_model",outputs_model);
           
            metric_knn = confusionMatrix(outputs_model, targets_test);
        
            println(" ");
            println("Metrics: ");
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

        for i in 1:k
            
            model = DecisionTreeClassifier(max_depth = modelHyperparameters["max_depth"], random_state = modelHyperparameters["random_state"]);

            d = modelHyperparameters["max_depth"];

            folder_path = "utils/ModelsTrained/modeldt $d";

            if !isdir(folder_path)
                mkpath(folder_path)
            end

            @assert((sum(crossValidationIndices .!= i)) > 1); 
        
            nEl = sum(crossValidationIndices .== i);
            n = size(inputs,1);
        
             inputs_training = inputs[(crossValidationIndices.!=i),:];
             targets_training = reshape(targets[(crossValidationIndices.!=i),:], sum(crossValidationIndices.!=i));
             inputs_test = inputs[(crossValidationIndices.==i),:];
             targets_test = reshape(targets[(crossValidationIndices.==i),:], sum(crossValidationIndices.==i));
 
  
            println(" ");
            println("------------------------------------------------------------------------------------------------------------------");
            println("-------------------------------------------------- FOLD: ", i, " -------------------------------------------------------");
            println("------------------------------------------------------------------------------------------------------------------");

            ScikitLearn.fit!(model, inputs_training, targets_training);

            outputs_model = ScikitLearn.predict(model, inputs_test);

            model1_file_path = joinpath(folder_path, "targets_dtmodel$i.jld2");
            JLD2.save(model1_file_path,"targets_test",targets_test);

            
            model2_file_path = joinpath(folder_path, "outputs_dtmodel$i.jld2");
            JLD2.save(model2_file_path,"outputs_model",outputs_model);

            metric_dt = confusionMatrix(outputs_model, targets_test);
                
            println(" ");
            println("Metrics: ");
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

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#                           UNIT 7                                         #
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#


function trainClassEnsemble(estimators::AbstractArray{Symbol,1}, 
    modelsHyperParameters:: AbstractArray{Dict, 1},     
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},    
    kFoldIndices:: Array{Int64,1})

    (inputs, targets) = trainingDataset;

    k = maximum(kFoldIndices); 

    ens_acc = Vector{Float64}(undef, k);
    ens_f1 = Vector{Float64}(undef, k);

    for i in 1:k


        models_dict = Dict()

        println("-------------------------------------------------------------------");
        println("-------------------------------$i Fold-----------------------------");
        println("-------------------------------------------------------------------");

        inputs_training = inputs[(kFoldIndices.!=i),:];
        targets_training = reshape(targets[(kFoldIndices.!=i),:], sum(kFoldIndices.!=i));
        inputs_test = inputs[(kFoldIndices.==i),:];
        targets_test = reshape(targets[(kFoldIndices.==i),:], sum(kFoldIndices.==i));

        @assert((sum(kFoldIndices .!= i)) > 1); 

        nEl = sum(kFoldIndices .== i);
        n = size(inputs,1);
        num_models = length(estimators)
        
        for j in 1:num_models
            if modelsHyperParameters[j]["model"] == "SVM"

                model_SVM = SVC(kernel = modelsHyperParameters[j]["kernel"], 
                degree = modelsHyperParameters[j]["kernelDegree"], 
                gamma = modelsHyperParameters[j]["kernelGamma"], 
                C = modelsHyperParameters[j]["C"],
                probability=true);
                
                models_dict["SVM $j"] = model_SVM;

            elseif modelsHyperParameters[j]["model"] == "kNN"

                model_kNN = KNeighborsClassifier(n_neighbors = modelsHyperParameters[j]["n_neighbors"]);
                
                models_dict["kNN $j"] = model_kNN;

            elseif modelsHyperParameters[j]["model"] =="DecisionTree"
            
                model_DT = DecisionTreeClassifier(max_depth = modelsHyperParameters[j]["max_depth"], random_state = modelsHyperParameters[j]["random_state"]);
                
                models_dict["DecisionTree $j"] = model_DT;
                
            else 
                println("Introducción del modelo erronea");
            end
        end 



        models_dict["Ensemble (Stacking)"] = StackingClassifier(estimators = [(name,models_dict[name]) for name in keys(models_dict)], final_estimator=SVC(probability=true), n_jobs=1);
        ScikitLearn.fit!(models_dict["Ensemble (Stacking)"], inputs_training, targets_training); 


        acc = ScikitLearn.score(models_dict["Ensemble (Stacking)"], inputs_test, targets_test);
        ens_acc[i] = acc;
        outputs_test = ScikitLearn.predict(models_dict["Ensemble (Stacking)"], inputs_test);

        folder_path = "utils/ModelsTrained/modelensemble";

        if !isdir(folder_path)
            mkdir(folder_path)
        end

        model1_file_path = joinpath(folder_path, "targets_ensemblemodel$i.jld2");
        JLD2.save(model1_file_path,"targets_test",targets_test);

        
        model2_file_path = joinpath(folder_path, "outputs_ensemblemodel$i.jld2");
        JLD2.save(model2_file_path,"outputs_model",outputs_test);


        metrics = confusionMatrix(outputs_test, targets_test);
        ens_f1[i] = metrics[7];

        println("Ensemble (Stacking): $(acc*100) %")
    
    end


    println("--------------------------------------------------------------------------------");
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