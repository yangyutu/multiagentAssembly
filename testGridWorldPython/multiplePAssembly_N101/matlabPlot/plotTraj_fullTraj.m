clear all
close all

addpath('')
%addpath(genpath('E:\Dropbox\matlabscript\'))
dataName = '../trajOutxyz_0';
data = load([dataName '.txt']);
N = max(data(:,1))+1;
np = N;
a=1e-6;
k=1.38e-23;
T=293.15;
dt = 1;
x = data(:,2);
y = data(:,3);

phi = data(:,4);
u = data(:,5);
nframe = length(x)/N;
mvflag = 0;

x= reshape(x,N,nframe);
y =reshape(y,N,nframe);
phi = reshape(phi,N,nframe);

u = reshape(u,N,nframe);

for i = 1:N
    figure(1)
    hold on
    plot(x(i,:), y(i,:))
end

xlim([-25, 25])
ylim([-25, 25])
figure(2)
plot(x(:,nframe), y(:,nframe),'o','markersize',14)

mvFlag = 0;
if mvFlag
    skip = 10;
    for j = 1:skip:nframe
     figure(2)
     hold on
       
    for i = 1:N
    figure(2)
    hold on
    plot(x(i,1:j), y(i,1:j),'linewidth',2)
    end
    
     xlim([0, 45])
    ylim([0, 45])
    saveas(gcf,[dataName '_' num2str(j) '.png'])
    close(2)
    end
end