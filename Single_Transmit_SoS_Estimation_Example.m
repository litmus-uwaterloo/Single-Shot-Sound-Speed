clear
%========================== Dataset Parameters
Nc=128; % Number of transducer channels
Ns=3120; % Number of samples per channel
fs=40e6; % Sampling Frequency
c0=1540; % 
rxdelay=4.e-6; % Sampling delay after transmission


%========================== Beamforming parameters
pitch=3.048e-4; % Transducer element pitch
pos_array_x= pitch*linspace(-(Nc-1)/2,(Nc-1)/2,Nc); % Transducer elements positions
tx_angle=-12; % Transmission angle (decimal)
Nz=128; % Number of rows in the beamforming grid
Nx=128; % Number of columns in the beamforming grid
pos_z = linspace(12e-3, 36e-3, Nz)'; % Beamforming grid rows locii
pos_x = linspace(-11e-3,9e-3, Nx); % Beamforming grid column locii
fnum = 1.4; % F number
delay_profile=(Nc-1:-1:0)*sind(abs(tx_angle))*pitch/c0; 

%========================== SoS estimation parameters
SoS_coarse=1460:20:1720; % SoS candidates to be evaluated
SoS_fine=1460:0.1:SoS_coarse(end); % SoS values to evaluate on the fitted function
total_phantoms=15; % Number of phantoms in DATAset
SoS_est=zeros(total_phantoms,1); % Pre allocate SoS estimates
SoS_ref=zeros(size(SoS_est)); % Pre allocate SoS reference values
pixel_percentage=0.8; % Percentage of pixels to use (ranked by intensity)


%========================== Filtering coeffs
filt_coeff = fir1(80,[4e6 7e6]/(fs/2)); % Transmitted frequency ~5MHz

%========================== Perform SoS estimatimation over the DATAset
figure,
for phantom_idx=1:1:total_phantoms
    % Load and filter DATA
    load(['DATA\Phantom' num2str(phantom_idx) '.mat'], 'rf_data', 'sos_meas');
    rf_data=hilbert(filtfilt(filt_coeff,1, rf_data(:,:)./max(rf_data(:,:),[],1)));

    % Scrap reference SoS values 
    SoS_ref(phantom_idx)=sos_meas;

    % Pre-allocate loss-values vector
    cross_channel_loss=zeros(length(SoS_coarse),1);

    % Obtain loss value for each SoS candidate
    for sos_idx=1:length(SoS_coarse)
        % Perform sampling through SoS-Aware ToF computation
        [sample_tensor]=SimplePseudoBeamformer(rf_data , pos_z , pos_x , pos_array_x , SoS_coarse(sos_idx) , tx_angle, fnum, fs ,rxdelay);

        % Rank and select the top intensity pixels for every SoS candidate
        sample_tensor=reshape(sample_tensor,[Nz*Nx,Nc]);
        insonified_pixels=sum(any(sample_tensor,2),"all");
        mean_intensity_img=mean(abs(sample_tensor),2, 'omitnan');
        [~,I]=maxk(mean_intensity_img(:), ceil(insonified_pixels*pixel_percentage));        
        sample_tensor=sample_tensor(I,:);
        sample_tensor(isnan(sample_tensor))=0;

        % Compute autocorrelations
        ffta = fft(sample_tensor,Nc,2);
        acv = ifft(ffta.*conj(ffta),[],2);

        % Compute negated SNACs as loss value
        cross_channel_loss(sos_idx)=-sum(abs(acv./max((acv),[],2)),"all");
    end
    % Fit a spline to the computed loss values for the SoS candidates
    fitted_loss = fit(SoS_coarse',cross_channel_loss,'smoothingspline','Normalize','on','SmoothingParam',0.99);
    
    % Plot spline
    plot(SoS_fine, fitted_loss(SoS_fine));
    hold on
    scatter(SoS_coarse,cross_channel_loss,"filled","square","k")
    xline(sos_meas)
    title(strcat("Phantom ",num2str(phantom_idx)))
    legend({'Fitted Loss','Evaluated SoS candidate'},'Location','northeast')
    xlabel("SoS")
    ylabel("Loss Value")
    hold off
    drawnow

    

    % Estimate global SoS based on minimum of fitted function
    [~,In] = min(fitted_loss(SoS_fine));
    SoS_est(phantom_idx) = SoS_fine(In);
    disp(['SoS Estimate: ' num2str(SoS_est(phantom_idx)) ' m/s      SoS Reference: ' num2str(sos_meas) ' m/s'])
end

%========================== Result reporting and plotting
considered_cases=(SoS_est~=SoS_coarse(1))&(SoS_est~=SoS_coarse(end));
est_errors=SoS_est(considered_cases)-SoS_ref(considered_cases);
disp(['Mean SoS Estimate Error: ' num2str(mean(est_errors)) ' m/s      Std: ' num2str(std(est_errors)) ' m/s'])
disp(['Percentage of Estimation Cases: ' num2str(sum(considered_cases)/total_phantoms*100) ' % '])

figure,
grid on
box on

scatter(SoS_ref,SoS_est,50,[0.6350 0.0780 0.1840],'^','filled','LineWidth',2);
axis equal tight

xlim([1480 1720]);
ylim([1480 1720]);
xticks(1480:40:1720)
yticks(1480:40:1720)
xlabel('SoS Reference Measurement')
ylabel('SoS Estimation')
title('Proposed Method')

hline = refline(1,0);
hline.LineStyle = '--';
hline.LineWidth = 2;
hline.Color = [0 0 0 0.25];
box on
grid on
