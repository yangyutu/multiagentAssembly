clear all
close all

nTarget = 7;
for i = 0 : nTarget
    data = load(['../testTraj_target_' num2str(i) '.txt']);
    dataSet{i+1} = data;
end


figure(1)
for i = 1 : nTarget+1
hold on
x = dataSet{i}(:,3);
y = dataSet{i}(:,4);

plot(y,x, 'linewidth',1)
end
xlim([0 35])
ylim([0 35])
set(gca, 'ydir','reverse')

%set(gca,'box','off')
%set(gca,'visible','off')
set(gca,'linewidth',2,'fontsize',20,'fontweight','bold','plotboxaspectratiomode','manual','xminortick','on','yminortick','on');
set(gca,'TickLength',[0.04;0.02]);
pbaspect([1 1 1])
%saveas(gcf,'traj.png')