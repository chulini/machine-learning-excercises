%octave ayuda:

(1 == 2) %false
(1 ~= 2) %true
xor(1,0) %true
PS('>> '); %cambia el principio
disp(sprintf('2 decimals: %0.2f',pi)) %imprime pi con 2 decimales: '2 decimals: 3.14'
format long %floats se ven con mas decimales
format short %floats se ven con mas decimales
ones(2,3) %matrix con puros 1's
eye(5) % identity matrix de 5x5
zeros(1,3) % matriz de 0's
rand(2,3) % matriz de randoms
randn(2,3) % matriz de normal randoms
hist(v) % histograma de vector v
hist(v,50) % histograma de vector v con 50 barritas

size(M) % devuelve matrix con valores de m y n
length(M) % devuelve el mayot entre m y n
pwd % prints curren path
cd 'path' %se moeve al path
ls
load ('featuresX.dat') %carga la matriz en featuresX.dat
who %muestra las variables actuales
whos %muestra las variables actuales con detalles
clear M %elimina la variable M
save hello.mat v %guarda la variable v en hello.mat en formato binario
save hello.mat v -ascii %guarda la variable v en hello.mat en formato ascii
A(3,2) %elemento 3,2 de la matrix A
A(2,:) % ":" significa todos los elementos en la fila/columna
A([1 3],:) % matriz con los elementos de la 1ra y la 3ra filas en ambas columnas
A(:,2) = [10; 11; 12] % reemplaza la segunda columna
A = [A, [100; 101; 102]] % agrega una columna a la derecha
A(:) % pone todos los elementos de la matriz en 1 solo vector
C = [A B] % concatena matrices A y B hacia al lado
C = [A; B] % concatena matrices A y B hacia abajo

A*C %multiplica matrices A y C
A .* B %multiplica usando el mismo indice
A .^ 2 %eleva todos los elementos de A al cuadrado
abs(v) %devuelve valor absoluto de vector v
A' %devuelve transpose de A
v < 3 %devuelve vector con true/false si cada elemento es < 3
magic(4) %devuelve cuadrado mágico de 4x4
sum(v) %suma valores del vector
prod(v) %multiplica valores del vector
floor(v) %revuelve matrix con floor de los elementos
ceil(v) %lo mismo con ceil
max(A,B) %Genera una matriz con el maximo entre cada par de valores de A y B
max(A,[],1) %devuelve matriz 1 fila con el maximo de cada columna
max(A,[],2) %devuelve matriz 1 col con el maximo de cada fila
max(max(A)) %Devuelve el valor maximo de toda la matriz
sum(A,1) %devuelve matriz 1 fila con la suma de cada columna
flipud(A) %flipea up down matriz A
pinv(A) %devuelve el inverso de A. pinv(A)*A = IDENTITY
y(1:1+10000,:) = []; %elimina las primeras 10001 filas
X(end-10000:end,:) = []; %elimina las ultimas 10001 filas

%FOR
for i=1:10,
	v(i) = 2^i;
end;

%WHILE
i = 1;
while true,
 v(i) = 999;
 i = i+1;
 if(i == 6),
   break;
  end;
end;

%PLOT:
graphics_toolkit gnuplot;
t = [0:0.01:0.98];
y1 = sin(2*pi*4*t);
plot(t,y1); %grafica x,y con lineas
y2 = cos(2*pi*4*t);
hold on; %no crea nuevo grafico
plot(t,y2,'r'); %'r' color
xlabel('time');
ylabel('value');
legend('sine','cosine');
title('sine vs cosine');
cd '~/Desktop/github/';
print -dpng 'myPlot.png'


close; %quita hold on


% data = load('ex1data1.txt');
% X = data(:, 1); y = data(:, 2);
% m = length(y); % number of training examples
% X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
% theta = zeros(2, 1); % initialize fitting parameters
% iterations = 1500;
% alpha = 0.01;
% computeCost(X, y, theta)
% h = X*theta;
% gradientDescent(X, y, theta, alpha, iterations)

[m, n] = size(X);
X = [ones(m, 1) X];

theta = zeros(n + 1, 1);
m = size(X,1);
h = sigmoid(X*theta);
J = -(1/m)*sum(y.*log(h) + (1-y).*log(1-h));

