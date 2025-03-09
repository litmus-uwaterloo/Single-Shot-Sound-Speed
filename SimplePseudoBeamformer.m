function [Imaged_tensor]=SimplePseudoBeamformer(rf_data , pos_z , pos_x , pos_array_x , sos_medium , tx_angle, Fnumber, fs ,RXdelay)
% size(pos_z)=[Nz,1];
% size(pos_x)=[1,Nx];

% Pre-allocate tensor that will contain the sampled values based on SoS
% corrected ToF
Imaged_tensor=inf(length(pos_z) , length(pos_x) , length(pos_array_x));

% Pre-allocate tensor that will contain all ToFs from each pixel to probe
% element
tof_tensor=zeros(size(Imaged_tensor));

% Pre-allocate tensor to use as flag for being inside the region defined by
% the Fnumber for each pixel and probe element
fnumber_tensor=false(size(Imaged_tensor));

% Perform computation of SoS-aware RX ToFs and Fnumber based masking
for chann_idx=1:1:length(pos_array_x)
    tof_tensor(:,:,chann_idx)=sqrt(pos_z.^2 + (pos_x-pos_array_x(chann_idx)).^2 ) / sos_medium;
    fnumber_tensor(:,:,chann_idx)=(abs(pos_z/Fnumber/2) - abs(pos_x-pos_array_x(chann_idx))) >= 0;
end

% Compute transmit tof only, by using snell's law at the interface
if tx_angle>0
    tx_angle=asin(sind(tx_angle));
    tx_times=(pos_z.*cos(tx_angle)+pos_x.*sin(tx_angle))/sos_medium;
    insonified_mask=(pos_z.*tan(tx_angle)-pos_x)<0;
else
    tx_angle=asin(sind(-tx_angle));
    tx_times=(pos_z.*cos(tx_angle)+(pos_array_x(end)-pos_x).*sin(tx_angle))/sos_medium;
    insonified_mask=(pos_z.*tan(tx_angle)-(pos_array_x(end)-pos_x))<0;
end

% Compute round ToF from each pixel to each probe element and remove the
% sampling delay
tx_times(~insonified_mask)=-inf;
tof_tensor=tof_tensor+tx_times-RXdelay;

% Set data limits to 0 to manage unsamplable ToFs
rf_data([1,end],:)=0;

% Fill the samples spots from the ToF values
for chann_idx=1:length(pos_array_x)
    % Do this procedure one channel at a time to minimize system requirements
    Imaged_tensor_channel=Imaged_tensor(:,:,chann_idx);
    tof_tensor_channel=tof_tensor(:,:,chann_idx);
    fnumber_tensor_channel=fnumber_tensor(:,:,chann_idx);

    % Fill in the sampled values from each channel, with interpolation
    Imaged_tensor_channel(fnumber_tensor_channel)=rf_data( min(max(floor(tof_tensor_channel(fnumber_tensor_channel)*fs),1),size(rf_data,1)), chann_idx).*(1-tof_tensor_channel(fnumber_tensor_channel)*fs+floor(tof_tensor_channel(fnumber_tensor_channel)*fs))+...
        rf_data( min(max(ceil(tof_tensor_channel(fnumber_tensor_channel)*fs),1),size(rf_data,1)), chann_idx).*(tof_tensor_channel(fnumber_tensor_channel)*fs-floor(tof_tensor_channel(fnumber_tensor_channel)*fs))   ;
    
    % Dump the values into the imaged tensor
    Imaged_tensor(:,:,chann_idx)=Imaged_tensor_channel;
end

% Replace infs for nans
Imaged_tensor(Imaged_tensor==inf)=nan;
end