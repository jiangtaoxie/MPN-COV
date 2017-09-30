function imdb = aircraft_get_database(aircraftDir, fgClass)
imdb.imageDir = fullfile(aircraftDir, 'data','images');

% Flexibly switch between various models
switch fgClass
    case 'family'
        labelSuffix = 'family';
    case 'manufacturer'
        labelSuffix = 'manufacturer';
    case 'variant'
        labelSuffix = 'variant';
    otherwise
        disp('Error: invalid class?');
end

% Training set
[imageNames, classLabels] = textread(fullfile(aircraftDir, 'data', ...
    sprintf('images_%s_train.txt', labelSuffix)), '%7s%*1s%s', 'delimiter', '\n', 'whitespace', '');

imdb.classes.name = unique(classLabels);
imdb.images.name = horzcat(imageNames(:));
[~,label] = ismember(classLabels, imdb.classes.name);
imdb.images.label = label';
imdb.images.set = ones(1, length(label));

% Val set
[imageNames, classLabels] = textread(fullfile(aircraftDir, 'data', ...
    sprintf('images_%s_val.txt', labelSuffix)), '%7s%*1s%s', 'delimiter', '\n', 'whitespace', '');
imdb.images.name = cat(1, imdb.images.name, horzcat(imageNames(:)));
[~,label] = ismember(classLabels, imdb.classes.name);
imdb.images.label = cat(2, imdb.images.label, label');
imdb.images.set = cat(2, imdb.images.set, 2*ones(1, length(label)));

% Test set
[imageNames, classLabels] = textread(fullfile(aircraftDir, 'data', ...
    sprintf('images_%s_test.txt', labelSuffix)), '%7s%*1s%s', 'delimiter', '\n', 'whitespace', '');
imdb.images.name = cat(1, imdb.images.name, horzcat(imageNames(:)));
[~,label] = ismember(classLabels, imdb.classes.name);
imdb.images.label = cat(2, imdb.images.label, label');
imdb.images.set = cat(2, imdb.images.set, 3*ones(1, length(label)));

% Append jpg
imdb.images.name = cellfun(@(x) [x '.jpg'], imdb.images.name, 'UniformOutput', false);

% Create ids
imdb.images.id = 1:numel(imdb.images.label);

% make this compatible with the OS imdb
imdb.meta.classes = imdb.classes.name ;
imdb.meta.inUse = true(1,numel(imdb.meta.classes)) ;
imdb.images.difficult = false(1, numel(imdb.images.id)) ; 
