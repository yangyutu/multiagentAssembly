clear all
close all

target = load('targetOrder.txt');
assignment = load('assignment.txt');

for i = 1 : size(target,1)
    numbers{i} = target(i,3);
end
figure(1)
x = target(:,1);
y = target(:,2);
plot(x, y, 'o','markersize', 14,'linewidth',1,'color','black')
text(x, y, numbers(1:length(x)), 'HorizontalAlignment','center', 'VerticalAlignment','middle')
axis([-20 20 -20 20])
pbaspect([1,1,1])
tx = target(:,1);
ty = target(:,2);

for i = 1 : size(target,1)
    numbers{i} = assignment(i,3);
end
figure(2)
x = assignment(:,1);
y = assignment(:,2);
plot(tx, ty, 'o','markersize',14,'linewidth',2,'color','black')
hold on
plot(x, y, 'o','markersize', 14)
text(tx, ty, numbers(1:length(x)), 'HorizontalAlignment','center', 'VerticalAlignment','middle')
text(x, y, numbers(1:length(x)), 'HorizontalAlignment','center', 'VerticalAlignment','middle')