clear all
close all hidden
clc
load('Fachada.mat')
hyperimg = rescale(hyperimg);
[M,N,L]=size(hyperimg);
Ho = [speye(M*N) ;sparse(M*(L-1),M*N)];
Co = diag(sparse(rand(M*(N+L-1),1)<0.5));

for l = 1:L
    HShift{l} = circshift(Ho,M*(l-1),1);    
end
HShift = blkdiag(HShift{:});
HMod = kron(speye(L),Co);

HInt = kron(ones(1,L),speye(M*N));
H = HInt*HShift'*HMod*HShift;

%% Paralelepipedo oblicuo
x1 = reshape(HShift*hyperimg(:),[M N+L-1 L]);
implay(x1)

%% Code Aperture sin shifting
x2 = reshape(HMod*x1(:), [M N+L-1 L]);
implay(x2)

%% Reorganización del tensor modulado
x3 = reshape(HShift'*x2(:) ,[M N L]);
implay(x3)

%% Medidas
Y = H*hyperimg(:);
Y_ = HInt*x3(:);
imagesc(rescale(reshape(Y,[M N])))
axis equal
xlabel(['Distancia entre Y y Y_ es nula: ' num2str(norm(Y-Y_))])

%%
save('H1ShotDriven','H')