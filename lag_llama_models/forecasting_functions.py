import numpy as np
from scipy.stats import pearsonr,spearmanr
import random

class probabilistic_forecasting():
    
    def sample_sorting(self,sample_data):     
        '''We need sorted samples to re-order accoeding to the ranks of the residuals
        i.e. we first sort the samples in ascending oder, with sorted sample it is easy to rank 
        
        parameters : 
        --------------- 
        sample_data : Probabilistic forecast samples using probabilistic forecasting methods'''
        
        sorted_samples = []
        for i in range(len(sample_data)):
            list_of_sorted_samples = []
            for j in range(len(sample_data[0])):
                ab = list(sample_data[i][j][0]) # sorting difference 
                ab.sort()
                list_of_sorted_samples.append(ab)
            sorted_samples.append(list_of_sorted_samples)
        return(sorted_samples)
    def sample_sorting_1(self,sample_data):
        sorted_samples = []
        for i in range(len(sample_data)):
            list_of_sorted_samples = []
            for j in range(len(sample_data[0])):
                ab = list(sample_data[i][j])
                ab.sort()
                list_of_sorted_samples.append(ab)
            sorted_samples.append(list_of_sorted_samples)
        return(sorted_samples)


    def sample_reordering(self,sample_data,ranks):
        '''We  rorder the future sample which is equal in size to the past data
        which is equal in size to ranks are. We have the ranks for the residual of the past data
        using rank rodering we are trying to preserve the rank correlation for the future values
        
        parameters : 
        ---------------  
            
        sample_data : Probabilistic forecast samples using probabilistic forecasting methods
        
        ranks : We defined the ranks on the standard residuals (difference(actual-mean(forecast)/std(forecast))  of the forecast 
        '''
        reorderd_samples =[]
        sorted_ = self.sample_sorting(sample_data)
        for i in range(len(sorted_)):
            pp = []
            for j in range(len(sorted_[i])):
                aa = []
                ab = ranks[i]-1
                for k in range(len(sorted_[i][j])):
                    aa.append(sorted_[i][j][int(ab[k])])
                pp.append((aa))
            reorderd_samples.append((pp))
        return(reorderd_samples)
    
    def sample_reordering_3(self,sample_data,ranks):
        '''We  rorder the future sample which is equal in size to the past data
        which is equal in size to ranks are. We have the ranks for the residual of the past data
        using rank rodering we are trying to preserve the rank correlation for the future values
        
        parameters : 
        ---------------  
            
        sample_data : Probabilistic forecast samples using probabilistic forecasting methods
        
        ranks : We defined the ranks on the standard residuals (difference(actual-mean(forecast)/std(forecast))  of the forecast 
        '''
        reorderd_samples =[]
        sorted_ = self.sample_sorting_1(sample_data)
        for i in range(len(sorted_)):
            pp = []
            for j in range(len(sorted_[i])):
                aa = []
                ab = ranks[i]-1
                for k in range(len(sorted_[i][j])):
                    aa.append(sorted_[i][j][int(ab[k])])
                pp.append((aa))
            reorderd_samples.append((pp))
        return(reorderd_samples)
    
    def sample_reordering_1(self,sample_data,ranks,rank_index):
        ''' Does the similar work as sample_reordering, defined due to output of time series forecast 
        
        parameters : 
        ---------------  
        sample_data : Probabilistic forecast samples using probabilistic forecasting methods
        
        ranks : We defined the ranks on the standard residuals (difference(actual-mean(forecast)/std(forecast))
                                                                of the forecast 

        rank_index: Indices of the ranks 

            '''        
        reorderd_samples =[]
        sorted_ = self.sample_sorting_1(sample_data)
        for i in range(len(sorted_)):
            pp = []
            for j in range(len(sorted_[i])):
                aa = []
                ab = ranks[rank_index[i]]-1
                for k in range(len(sorted_[i][j])):
                    aa.append(sorted_[i][j][int(ab[k])])
                pp.append((aa))
            reorderd_samples.append((pp))
        return(reorderd_samples)
    
    def getting_bottom_series(self, sample_data,sum_mat,sum_mat_labels):        
        '''Function for getting bottom level series  
        parameters : 
        ---------------  
        
        sample_data : Probabilistic forecast samples using probabilistic forecasting methods
        
        sum_mat : Summing matrix defined on the hierarchy of the data 
        
        sum_mat_labels: Columns of the time series data
        

        '''
        bottom_s = []
        BU_len = sum_mat.shape[1]   #length of bottom series
        bottom_samples_index = list(range(len(sum_mat_labels)-BU_len,len(sample_data)))
        for i in bottom_samples_index:
            bottom_s.append(sample_data[i])
        return(bottom_s)
    
    
    def adding_list(self,list_):
        '''Adds the list element wise
        
        parameters : 
        ---------------  
        
        list_ : List od lists 
        '''
        abc = [sum(x) for x in zip(*list_)]
        return abc
    
    def list_reversal(self,list_):
        '''reverse the order of the list
        
        parameters : 
        ---------------  
        
        list_ : list of lists
        
        '''
        aa = []
        for i in range(len(list_)-1,-1,-1):
            aa.append(list_[i])
        return aa

    def bottom_up_forecast(self, columns,bottom_series_,bottom_level,aggre_series_numbers,levels):
        '''This method gives the forecast using bottom up method by hyndman et.al.[1]. 
        by bottom up method we keep the forecast for bottom levels as it is and for aggregated levels we 
        add the child node to get the foorecast for the parent node.
        
        parameters : 
        ---------------  
        
       
        columns :List column names of the data
        
        bottom_series_ : Bottom series of the hierarchical data
        
        bottom_level:  Indeices of the bottom level
        
        aggre_series_numbers: Indices of the child nodes for each aggregated series 
        
        levels : indices of series at each level(e.g. suppose there 3 series at level first then, levels[1] = [0,1,2])
        
        -------------
        
        References
        ------------
        1. Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2018.'''
        
        bottom_up_ =[]
        for i in range(len(columns)-1,-1,-1):
            bb= []
            if i in bottom_level :
                for j in range(len(bottom_series_[0])): 
                    aa = []
                    for k in aggre_series_numbers[i]:
                        k = int(k)
                        aa.append(bottom_series_[k][j])
                    ab = self.adding_list(aa)
                    #print(aa)
                    bb.append(np.array(ab))
                    aa = [] 
                bottom_up_.append(np.array(bb))
                bb=[]
        # above steps justs adds the bottom level series to the bottom_up_ forecast list 
        # steps below are to get the forecast for aggregated levels.
        len_of_agg = len(bottom_up_)
        for l in range(len(levels)-2,-1,-1):
            cc  = bottom_up_[-len(levels[l+1]):]
            cc1 = self.list_reversal(cc)
            bottom_bb = []
            for i in range(len(columns)-1-len_of_agg,-1,-1):
                bb = []
                if i in levels[l]:
                    for j in range(len(cc1[0])):
                        aa = []
                        for k in aggre_series_numbers[i]:
                            k = int(k)
                            aa.append(cc1[k][j])  
                        ab = self.adding_list(aa)
                        bb.append(np.array(ab))
                        aa= []
                    bottom_bb.append(np.array(bb))
                    bb=[]
            bottom_up_= bottom_up_+ bottom_bb
            bottom_bb =[]
            cc= []
            cc1=[]
        return(bottom_up_)
    def bottom_up_revised_forecast(self,columns,bottom_series_r,bottom_level,aggre_series_numbers,levels,ranks):
        '''This method gives the bottom up forecast with reodering [1]. Bottom up method adds the childe node series as it is 
        this method preserves the correlation between child node series by reodering the forecated values according the 
        rannks defined on the standard reisduals defined on the predicted values of the past data.
        
        parameters : 
        ---------------  
        columns :List column names of the data
        
        bottom_series_r: bottom series of the hiearchy but reoder acccording to the ranks of the residuals
        
        bottom_level:  Indeices of the bottom level
        
        aggre_series_numbers: Indices of the child nodes for each aggregated series 
        
        levels : indices of series at each level(e.g. suppose there 3 series at level first then, levels[1] = [0,1,2])
        
        
             
        References
        ------------
        
        1. Taieb, Souhaib Ben, James W. Taylor, and Rob J. Hyndman. "Coherent probabilistic forecasts for hierarchical time series." 
        International Conference on Machine Learning. PMLR, 2017.
        '''
        
        revised_samples=[]
        for i in range(len(columns)-1,-1,-1):
            bb= []
            if i in bottom_level :
                for j in range(len(bottom_series_r[0])):
                    aa = []
                    for k in aggre_series_numbers[i]:
                        k = int(k)
                        aa.append(bottom_series_r[k][j])
                    ab = self.adding_list(aa)
                    bb.append(np.array(ab))
                    aa = [] 
                revised_samples.append(np.array(bb))
                bb=[]
        len_of_agg = len(revised_samples)
        for l in range(len(levels)-2,-1,-1):
                cc  = revised_samples[-len(levels[l+1]):]
                ranks_index = levels[l+1]              #rank index are the actual index of the time series
                cc1 = self.list_reversal(cc)
                cc1= self.sample_reordering_1(cc1,ranks,ranks_index)
                bottom_bb_r = []
                for i in range(len(columns)-1-len_of_agg,-1,-1):
                    bb = []
                    if i in levels[l]:
                        for j in range(len(cc1[0])):
                            aa = []
                            for k in aggre_series_numbers[i]:
                                k = int(k)
                                aa.append(cc1[k][j])  
                            ab = self.adding_list(aa)
                            bb.append(np.array(ab))
                            aa= []
                        bottom_bb_r.append(np.array(bb))
                        bb=[]
                revised_samples = revised_samples+ bottom_bb_r
                bottom_bb_r =[]
                cc= []
                cc1=[]
        return(revised_samples)


    def find_corr(self,series1,series2):
        '''Calculated the correlation between two sereis
        
        parameters : 
        --------------- 
        
        series1, series2 : lists (time series converted into list)
        
        '''
        
        c,p = spearmanr(series1,series2)
        return(c)


    def min_max_n_elements(self,lst, n):
        ''' Find the n smallest and largest (outliers) elements with their indices 
        
        parameters : 
        --------------- 
        
        lst : List
        n : number of outliers/2   
        '''
        
        
        min_elements = sorted([(val, idx) for (idx, val) in enumerate(lst)])[:n]
        max_elements = sorted([(val, idx) for (idx, val) in enumerate(lst)], reverse=True)[:n]
    
        # Extract the values and indices from the tuples
        min_values = [val for (val, idx) in min_elements]
        min_indices = [idx for (val, idx) in min_elements]
        max_values = [val for (val, idx) in max_elements]
        max_indices = [idx for (val, idx) in max_elements]
    
        # Return the result as two lists of values and two lists of indices
        return min_values+max_values, min_indices+ max_indices

    def maximize_correlation_large(self,l1,list_f,nc_n, ten_series,corr_past, current_diff,outliers,outlier_indices):
        
        '''Function re-orders the outlier elements of the each of these three series. For reoedering we have past correlation, and each
        outlier is order in a way such that the difference between past correlation and current correlation by re odering is minimum.
        correlation 
        
        parameter : 
        -----------
        l1 : forecasted values of  time series in list format 
        
        corr_past: past correlation between the time series 
        
        current_diff: difference between the forecasted and past values of the series before reodering 
        
        
        outliers: We took n  minimum and n maximum values of the given series and called it as outliers 
        
        outlier_indices = indices of the outliers are store for reodering 
        
        
        #rember to change correlation past
        
        '''
        
        
        list1 = l1.copy()
        best_diff = current_diff
        best_list1 = list1
        list1_og_indices =  []
        for n in range(len(list1)):
            list1_og_indices.append(n)
        outlier_reorderd_index = []
        for j in range(len(outliers)):
            for i in range(len(list1)):
                if i not in outlier_reorderd_index:  #if index of the current outlier not in reodered index 
                    list1[list1_og_indices.index(outlier_indices[j])], list1[i] = list1[i],  list1[list1_og_indices.index(outlier_indices[j])]
                    corr_future_ =[]
                    for ik in range(nc_n):
                        corr_f_ = self.find_corr(list1,list_f[ten_series[ik]])
                        corr_future_.append(corr_f_)
                    diff_1 = [abs(corr_future_[ik2]- corr_past[ik2]) for ik2 in range(len(corr_past))]
                    diff_ = np.sum(diff_1)
                    if round(diff_, 4) < best_diff:
                        best_diff = diff_
                        best_list1 = list1.copy()
                        best_index = i
                    else:
                        best_index = list1_og_indices.index(outlier_indices[j])
                    list1[list1_og_indices.index(outlier_indices[j])], list1[i] = list1[i],  list1[list1_og_indices.index(outlier_indices[j])]
            outlier_reorderd_index.append(best_index) #store the ondex of the reordered outlier
            list1[best_index],list1[list1_og_indices.index(outlier_indices[j])] = outliers[j],list1[best_index] #final best updated is reordered in the list s
            list1_og_indices[best_index], list1_og_indices[list1_og_indices.index(outlier_indices[j])]

        return best_list1
    
    

    def reordering_series(self,list_f,list_f_p,outlier_frac):
        '''The function runs over all the child nodes and, at a time, takes single time series check the correlation of that times series with other randomly   selected 
        time series and reorders the time series outliers such that it is close to the correlation between past values of this three-time series
        
        parameters
        --------------
        list_f : forecasted timse sereis list (contains child node forecast for each series)
        list_f_p : past values for the time series in list_f
        
        '''
        correlated_series = []
        for i in range(len(list_f)):
            ''' below we are taking the two time series other than aggregated time series i of the same parents 
          to take the correlation between them '''
            if len(list_f) > 1:
                oth_series = list(range(len(list_f)))
                oth_series.remove(i)
                nc_n = len(oth_series) #number of neibour series to consider 
                 #here we are taking the outlier of each series to reorder them by considering the correlation  
                outliers, outlier_indices = self.min_max_n_elements(list_f[i],int(len(list_f[0])*outlier_frac))
                corr_past =[]
                corr_future =[]
                for ik in range(nc_n):
                     corr_f = self.find_corr(list_f[i],list_f[oth_series[ik]])
                     corr_p = self.find_corr(list_f_p[i],list_f_p[oth_series[ik]])
                     corr_future.append(corr_f)
                     corr_past.append(corr_p)
                current_diff_list = [abs(corr_future[ik1]- corr_past[ik1]) for ik1 in range(len(corr_past))]
                current_diff = np.sum(current_diff_list)
                optimal_order = self.maximize_correlation_large(list_f[i],list_f,nc_n,oth_series, corr_past,current_diff,outliers,outlier_indices )
                correlated_series.append(optimal_order)

                list_f[i] = optimal_order
    
            else:
                correlated_series.append(list_f[i])
                list_f[i] = list_f[i]
        return(correlated_series)
    
    
    def bottom_up_forecast_hueristic(self, bottom_series_, bottom_series_og,columns,bottom_level,aggre_series_numbers,levels,outlier_frac):
        ''' This method gives a bottom-up probabilistic forecast but with outlier reordering. It takes the p% outlier from each time step 
        forecasted samples, randomly selects the other two series from the child nodes of the same parent and then reorders those outliers 
        for each series to preserve the correlation of past values and forecasted values.
        
        
        parameters
        ----------------------
        
        bottom_series_ : Bottom series of the hierarchical data for the forecasted values
        
        bottom_series_og: Bottom series of the hierarchical past data
        
        columns :List column names of the data
                  
         bottom_level:  Indeices of the bottom level
         
         aggre_series_numbers: Indices of the child nodes for each aggregated series 
         
         levels : indices of series at each level(e.g. suppose there 3 series at level first then, levels[1] = [0,1,2])'''
        
        bottom_up_hu =[]
        bottom_up_og = []
        for i in range(len(columns)-1,-1,-1):    # bottom level forecast 
            bb= []
            bb_og = []
            if i in bottom_level :
                aa_og = []
                for k1 in aggre_series_numbers[i]:
                    k1 = int(k1)
                    aa_og.append(bottom_series_og[k1])
                ab_og = self.adding_list(aa_og)
                bb_og.append(np.array(ab_og))
                aa_og = []
                bottom_up_og.append(np.array(bb_og))
                bb_og = []
                for j in range(len(bottom_series_[0])):
                    aa = []
                    for k in aggre_series_numbers[i]:
                        k = int(k)
                        aa.append(bottom_series_[k][j])  
                    ab = self.adding_list(aa)
                    bb.append(np.array(ab))
                    aa = [] 
                bottom_up_hu.append(np.array(bb))
                bb=[]
                
        len_of_agg = len(bottom_up_hu)   #number of bottom level series
        for l in range(len(levels)-2,-1,-1):     # there are three levels 
            cc  = bottom_up_hu[-len(levels[l+1]):]
            cc_og = bottom_up_og[-len(levels[l+1]):]  #this is the bottom series 
            cc1_og = self.list_reversal(cc_og)     # earlier series was in reverse order
            cc1 = self.list_reversal(cc)
            bottom_bb = []
            bottom_bb_og = []         
            for i1 in range(len(columns)-1-len_of_agg,-1,-1):
                bb = []
                bb_og = []
                if i1 in levels[l]:
                    aa_og = []
                    for k2 in aggre_series_numbers[i1]:
                        k2 = int(k2)
                        aa_og.append(cc1_og[k2])
                    ab_og = self.adding_list(aa_og)
                    bb_og.append(np.array(ab_og))
                    aa_og_l = []
                    for s in range(len(aa_og)):
                        aa_og_l.append(list(aa_og[s][0]))
                    for j1 in range(len(cc1[0])):
                        aa = []
                        for k3 in aggre_series_numbers[i1]:
                            k3 = int(k3)
                            aa.append(cc1[k3][j1])               #child nodes for aggregated series
                        aa = self.reordering_series(aa,aa_og_l,outlier_frac)
                        for i in range(5):           #why i took 5 here , this is done to reorder it multiple times
                            aa = self.reordering_series(aa,aa_og_l,outlier_frac)
                        ab = self.adding_list(aa)
                        bb.append(np.array(ab))
                        aa= []
                    bottom_bb.append(np.array(bb))
                    bb=[]                 
                    bottom_bb_og.append(np.array(ab_og))
                    bb_og =[]
            bottom_up_hu= bottom_up_hu+ bottom_bb
            bottom_up_og = bottom_up_og + bottom_bb_og
            bottom_bb =[]
            bottom_bb_og = []
            cc= []
            cc_og = []
            cc1=[]
            cc1_og = []
        return(bottom_up_hu)

    def finding_mean(self,sample):
        ''' function to define the mean of the sanple forecats'''
        
        mean_sample = []
        for i in range(len(sample)):
            aa = []
            for j in range(len(sample[i])):
                aa.append(np.mean(sample[i][j]))
            mean_sample.append(aa)
            aa =[]
        return(mean_sample)














