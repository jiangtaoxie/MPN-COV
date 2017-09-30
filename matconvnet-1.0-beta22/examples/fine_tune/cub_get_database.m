function imdb = cub_get_database(cubDir, useCropped, useVal)
% Automatically change directories
% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).
if useCropped
    imdb.imageDir = fullfile(cubDir, 'images_cropped') ;
else
    imdb.imageDir = fullfile(cubDir, 'images');
end

imdb.maskDir = fullfile(cubDir, 'masks'); % doesn't exist
imdb.sets = {'train', 'val', 'test'};

% Class names
[~, classNames] = textread(fullfile(cubDir, 'classes.txt'), '%d %s');
imdb.classes.name = horzcat(classNames(:));

% Image names
[~, imageNames] = textread(fullfile(cubDir, 'images.txt'), '%d %s');
imdb.images.name = imageNames;
imdb.images.id = (1:numel(imdb.images.name));

% Class labels
[~, classLabel] = textread(fullfile(cubDir, 'image_class_labels.txt'), '%d %d');
imdb.images.label = reshape(classLabel, 1, numel(classLabel));

% Bounding boxes
[~,x, y, w, h] = textread(fullfile(cubDir, 'bounding_boxes.txt'), '%d %f %f %f %f');
imdb.images.bounds = round([x y x+w-1 y+h-1]');

% Image sets
[~, imageSet] = textread(fullfile(cubDir, 'train_test_split.txt'), '%d %d');
imdb.images.set = zeros(1,length(imdb.images.id));
imdb.images.set(imageSet == 1) = 1;
imdb.images.set(imageSet == 0) = 3;

if useVal
    rng(0)
    trainSize = numel(find(imageSet==1));
    
    trainIdx = find(imageSet==1);
    
    % set 1/3 of train set to validation
    valIdx = trainIdx(randperm(trainSize, round(trainSize/3)));
    imdb.images.set(valIdx) = 2;
end

% make this compatible with the OS imdb
imdb.meta.classes = imdb.classes.name ;
imdb.meta.inUse = true(1,numel(imdb.meta.classes)) ;
imdb.images.difficult = false(1, numel(imdb.images.id)) ; 
