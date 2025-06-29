#include "buffer_pool_manager.h"

/**
 * @brief 从free_list或replacer中得到可淘汰帧页的 *frame_id
 * @param frame_id 帧页id指针,返回成功找到的可替换帧id
 * @return true: 可替换帧查找成功 , false: 可替换帧查找失败
 */
bool BufferPoolManager::FindVictimPage(frame_id_t *frame_id) {
    // Todo:
    // 1 使用BufferPoolManager::free_list_判断缓冲池是否已满需要淘汰页面
    // 1.1 未满获得frame
    // 1.2 已满使用lru_replacer中的方法选择淘汰页面



    return false;
}

/**
 * @brief 更新页面数据, 为脏页则需写入磁盘，更新page元数据(data, is_dirty, page_id)和page table
 *
 * @param page 写回页指针
 * @param new_page_id 写回页新page_id
 * @param new_frame_id 写回页新帧frame_id
 */
void BufferPoolManager::UpdatePage(Page *page, PageId new_page_id, frame_id_t new_frame_id) {
    // Todo:
    // 1 如果是脏页，写回磁盘，并且把dirty置为false
    // 2 更新page table
    // 3 重置page的data，更新page id

    // 1 如果是脏页，写回磁盘，并且把dirty置为false
     

    // 2 更新page table
    //这里有个疑问：page在pages_数组中的位置是固定的，它的frame_id难道不应该是固定的吗？为什么还能改成新frame_id?
     

    // 3 重置page的data，更新page id

    
}

/**
 * Fetch the requested page from the buffer pool.
 * 如果页表中存在page_id（说明该page在缓冲池中），并且pin_count++。
 * 如果页表不存在page_id（说明该page在磁盘中），则找缓冲池victim page，将其替换为磁盘中读取的page，pin_count置1。
 * @param page_id id of page to be fetched
 * @return the requested page
 */
Page *BufferPoolManager::FetchPage(PageId page_id) {
    // Todo:
    // 0.     lock latch
    // 1.     Search the page table for the requested page (P).
    // 1.1    If P exists, pin it and return it immediately.
    // 1.2    If P does not exist, find a replacement page (R) from either the free list or the replacer.
    //        Note that pages are always found from the free list first.
    // 2.     If R is dirty, write it back to the disk.
    // 3.     Delete R from the page table and insert P.
    // 4.     Update P's metadata, read in the page content from disk, and then return a pointer to P.
    std::scoped_lock lock{latch_};

    Page *targetPage = nullptr;
    //Todo:
    // 1.     从page_table_中搜寻目标页
    // 1.1    若目标页有被page_table_记录，则将其所在frame固定(pin)，并返回目标页。
    if(page_table_.count(page_id) > 0){
        frame_id_t fid = page_table_[page_id];
        replacer_->Pin(fid);

        //返回目标页
        targetPage = &(pages_[fid]);
        targetPage->pin_count_++;

        return targetPage;
    }
    // 1.2    否则，尝试调用find_victim_page获得一个可用的frame，若失败则返回nullptr
    else{
        //为了防止并发冲突，将find_victim_page展开
        frame_id_t fid(INVALID_FRAME_ID);
        bool found = false;
        // 1 使用BufferPoolManager::free_list_判断缓冲池是否已满需要淘汰页面
        // 1.1 未满获得frame
        bool is_empty = free_list_.empty();
        if(is_empty == false){
            fid = free_list_.front();
            free_list_.pop_front();
            Page *p = &pages_[fid];
            found = true;
        }
        else{
            // 1.2 已满使用lru_replacer中的方法选择淘汰页面
            found = replacer_->Victim(&fid);
        }
        if(found == false){
            return nullptr;
        }
        targetPage = &(pages_[fid]);

        // 2.调用updata_page修改页面信息，并在旧页面脏的情况下将其写回磁盘
        //原本此处调用update_page()函数，但调用需要临时解锁操作，会带来并发冲突问题，因此将update_page()剥离出来
        // 2.1 如果是脏页，写回磁盘，并且把dirty置为false
        if(targetPage->IsDirty()){
            //原本此处调用flush_page()函数，但调用需要临时解锁操作，会带来并发冲突问题
            //为了解决此问题，将flush_page()函数展开
            PageId old_page_id = targetPage->GetPageId();
            if(page_table_.count(old_page_id) > 0){
                frame_id_t fid = page_table_[old_page_id];
                disk_manager_->write_page(old_page_id.fd, old_page_id.page_no, targetPage->data_,PAGE_SIZE);   
            }
            targetPage->is_dirty_ = false;
        }
        // 2.2 更新page table
        page_table_.erase(targetPage->GetPageId());
        page_table_.insert(std::make_pair(page_id,fid));
        // 2.3 重置page的data，更新page id
        targetPage->ResetMemory();
        targetPage->id_ = page_id;

        // 3.调用disk_manager_的read_page读取目标页到frame
        disk_manager_->read_page(page_id.fd, page_id.page_no, targetPage->data_, PAGE_SIZE);

        // 4.固定目标页，更新pin_count_
        replacer_->Pin(fid);
        targetPage->pin_count_ = 1;

        // 5.返回目标页
        return targetPage;
    }
    return nullptr;



/* 
    // 0.     lock latch
    std::scoped_lock lock{latch_};

    // 1.     Search the page table for the requested page (P).
    std::unordered_map<PageId, frame_id_t, PageIdHash>::iterator page_it = page_table_.find(page_id);
    // 1.1    If P exists, pin it and return it immediately.
    if(page_it != page_table_.end()){
        frame_id_t fid = page_it->second;
        replacer_->Pin(fid);
        pages_[fid].pin_count_++;
        return &pages_[fid];
    } else { 
    // 1.2    If P does not exist, find a replacement page (R) from either the free list or the replacer.
    //        Note that pages are always found from the free list first.
        frame_id_t fid = INVALID_FRAME_ID;
        if(!FindVictimPage(&fid))
            return nullptr;
        Page *page = &pages_[fid];
        // if(page->is_dirty()) 加入这句话会报错！！！
        if (page->IsDirty()) 
           UpdatePage(page, page_id, fid);
        disk_manager_->read_page(page_id.fd, page_id.page_no, page->GetData(), PAGE_SIZE);
        replacer_->Pin(fid);
        page->pin_count_ = 1;
        return page;

 */

/* 
        frame_id_t fid = INVALID_FRAME_ID;
        if (!FindVictimPage(&fid)) return nullptr;

    // 2.     If R is dirty, write it back to the disk.
        Page *page = &pages_[fid];
        // if(page->is_dirty()) 加入这句话会报错！！！
        if( page->IsDirty()){
            UpdatePage(page, page_id, fid);
            //disk_manager_->write_page(page->GetPageId().fd, page->GetPageId().page_no, page->GetData(), PAGE_SIZE);
        }    
    // 3.     Delete R from the page table and insert P.
        page_table_.erase(page->GetPageId());
        page_table_[page_id] = fid;
 
     // 4.     Update P's metadata, read in the page content from disk, and then return a pointer to P.
        Page *newPage = page;
        disk_manager_->read_page(page_id.fd, page_id.page_no, newPage->data_, PAGE_SIZE);
        newPage->id_= page_id;
        newPage->pin_count_ = 1;
        newPage->is_dirty_ = false;
        replacer_->Pin(fid);

        return newPage; 
    } */
    //return nullptr;
}

/**
 * Unpin the target page from the buffer pool. 取消固定pin_count>0的在缓冲池中的page
 * @param page_id id of page to be unpinned
 * @param is_dirty true if the page should be marked as dirty, false otherwise
 * @return false if the page pin count is <= 0 before this call, true otherwise
 */
bool BufferPoolManager::UnpinPage(PageId page_id, bool is_dirty) {
    // Todo:
    // 0. lock latch
    // 1. try to search page_id page P in page_table_
    // 1.1 P在页表中不存在 return false
    // 1.2 P在页表中存在 如何解除一次固定(pin_count)
    // 2. 页面是否需要置脏

    std::scoped_lock lock{latch_};
    // 1. 尝试在page_table_中搜寻page_id对应的页P
    // 1.1 P在页表中不存在 return false
    if( page_table_.count(page_id) == 0) return false;
 
    // 1.2 P在页表中存在,获取其pin_count_
    frame_id_t fid = page_table_[page_id];
    Page *targetPage = &(pages_[fid]);    

    // 2.1 若pin_count_已经等于0,则返回false
    if(targetPage->pin_count_ == 0) return false;

    // 2.2 若pin_count_大于0，则pin_count_自减一
    if(targetPage->pin_count_>0){
        targetPage->pin_count_--;
        // 2.2.1 若自减后等于0，则调用replacer_的Unpin
        if(targetPage->pin_count_ == 0){
            replacer_->Unpin(fid);
        }
    }
    // 3 根据参数is_dirty，更改P的is_dirty_
    if (is_dirty == true){
        targetPage->is_dirty_ = true;
    }

    return true;


/*     std::scoped_lock lock{latch_};
    std::unordered_map<PageId, frame_id_t, PageIdHash>::iterator page_it = page_table_.find(page_id);
    if(page_it == page_table_.end())  return false;
    
    Page *P = &pages_[page_it->second];
    if(P->pin_count_ == 0)
        return false;
    P->pin_count_--;
    if(P->pin_count_ == 0)
        replacer_->Unpin(page_it->second);
    P->is_dirty_ = is_dirty || P->IsDirty();
    // P->is_dirty_ = is_dirty;
    return true;
 */


    /* std::scoped_lock lock{latch_};
    frame_id_t *frame_id = new frame_id_t;
    // if p exists
    if (page_table_.find(page_id) != page_table_.end()) {
        *frame_id = page_table_[page_id];
        if (pages_[*frame_id].pin_count_ > 0) {
            pages_[*frame_id].pin_count_--;
        }
        if (pages_[*frame_id].pin_count_ == 0) {
            
            // disk_manager_->write_page(pages_[*frame_id].GetPageId().fd, pages_[*frame_id].GetPageId().page_no, pages_[*frame_id].GetData(), PAGE_SIZE);
            replacer_->Unpin(*frame_id);
            
        }
        if (is_dirty == true) {
                pages_[*frame_id].is_dirty_ = true;
            }
        return true;
    } else
    
    return false; */
    //return true;
}

/**
 * Flushes the target page to disk. 将page写入磁盘；不考虑pin_count
 * @param page_id id of page to be flushed, cannot be INVALID_PAGE_ID
 * @return false if the page could not be found in the page table, true otherwise
 */
bool BufferPoolManager::FlushPage(PageId page_id) {
    // Todo:
    // 0. lock latch
    // 1. 页表查找
    // 2. 存在时如何写回磁盘
    // 3. 写回后页面的脏位
    // Make sure you call DiskManager::WritePage!
    std::scoped_lock lock{latch_};

    // 1. 查找页表,尝试获取目标页P
    // 1.1 目标页P没有被page_table_记录 ，返回false
    if(page_table_.count(page_id) == 0){
        return false;
    }
    if(page_id.page_no == -1){
        std::cout<<"问题发生,page_id.pag_no=-1"<<std::endl;
    }
    frame_id_t fid = page_table_[page_id];
    Page *page = &(pages_[fid]);
    // 2. 无论P是否为脏都将其写回磁盘。
    disk_manager_->write_page(page_id.fd, page_id.page_no, page->data_,PAGE_SIZE);

    // 3. 更新P的is_dirty_
    page->is_dirty_ = false;

    return true;


/*     std::scoped_lock lock{latch_};
    std::unordered_map<PageId, frame_id_t, PageIdHash>::iterator page_it = page_table_.find(page_id);
    if(page_it == page_table_.end())   return false;

    Page *p = &pages_[page_it->second];
    disk_manager_->write_page(p->GetPageId().fd, p->GetPageId().page_no, p->GetData(), PAGE_SIZE);
    p->is_dirty_ = false;
    return true;
 */

   /*  std::scoped_lock lock{latch_};
    for (size_t i = 0; i < pool_size_; i++) {
        Page *page = &pages_[i];
        if (page->GetPageId().page_no != INVALID_PAGE_ID && page->GetPageId() == page_id) {
            disk_manager_->write_page(page->GetPageId().fd, page->GetPageId().page_no, page->GetData(), PAGE_SIZE);
            page->is_dirty_ = false;
            return true;
        }
    }
    return false; */
    //return true;
}

/**
 * Creates a new page in the buffer pool. 相当于从磁盘中移动一个新建的空page到缓冲池某个位置
 * @param[out] page_id id of created page
 * @return nullptr if no new pages could be created, otherwise pointer to new page
 */
Page *BufferPoolManager::NewPage(PageId *page_id) {
    // Todo:
    // 0.   lock latch
    // 1.   Make sure you call DiskManager::AllocatePage!
    // 2.   If all the pages in the buffer pool are pinned, return nullptr.
    // 3.   Pick a victim page P from either the free list or the replacer. Always pick from the free list first.
    // 4.   Update P's metadata, zero out memory and add P to the page table. pin_count set to 1.
    // 5.   Set the page ID output parameter. Return a pointer to P.
    std::scoped_lock lock{latch_};

    frame_id_t fid(INVALID_FRAME_ID);
    // 1.   获得一个可用的frame，若无法获得则返回nullptr
    bool found = false;
    //为了防止并发冲突，将find_victim_page展开
    // 1.1 使用BufferPoolManager::free_list_判断缓冲池是否已满需要淘汰页面
    // 1.2 未满获得frame
    bool is_empty = free_list_.empty();
    if(is_empty == false){
        fid = free_list_.front();
        free_list_.pop_front();
        Page *p = &pages_[fid];
        found = true;
    }
    else{
        // 1.3 已满使用lru_replacer中的方法选择淘汰页面
        found = replacer_->Victim(&fid);
    }
    if(found == false){
        return nullptr;
    } 
    // 2.   在fd对应的文件分配一个新的page_id
    page_id->page_no = disk_manager_->AllocatePage(page_id->fd);
    
    // 3.   将frame的数据写回磁盘
    Page *page = &(pages_[fid]);
    if(page->IsDirty() == true){
        //原本此处调用flush_page()函数，但调用需要临时解锁操作，会带来并发冲突问题
        //为了解决此问题，将flush_page()函数剥离出来
        PageId old_page_id = page->id_;
        if(page_table_.count(old_page_id) > 0){
            frame_id_t fid = page_table_[old_page_id];
            Page *page = &(pages_[fid]);
            disk_manager_->write_page(old_page_id.fd, old_page_id.page_no, page->data_,PAGE_SIZE);
            page->is_dirty_ = false;
        }
    }

    // 4.   更新page_table_
    page_table_.erase(page->id_);
    page_table_.insert(std::make_pair(*page_id, fid));

    // 5.   固定frame，更新pin_count_
    replacer_->Pin(fid);
    page->pin_count_ = 1;

    // 6.   更新page_id
    page->id_.fd =  page_id->fd;
    page->id_.page_no = page_id->page_no;

    //7.    修改其元数据
    page->ResetMemory();

    // 8.   返回获得的page
    return page;

/*     std::scoped_lock lock{latch_};

    frame_id_t fid = INVALID_FRAME_ID;
    if(!FindVictimPage(&fid)) 	return nullptr;

    page_id->page_no = disk_manager_->AllocatePage(page_id->fd);
    Page *page = &pages_[fid];  
    UpdatePage(page, *page_id, fid);
    replacer_->Pin(fid);
    page->pin_count_ = 1;
    return page;
 */}

/**
 * @brief Deletes a page from the buffer pool.
 * @param page_id id of page to be deleted
 * @return false if the page exists but could not be deleted, true if the page didn't exist or deletion succeeded
 */
bool BufferPoolManager::DeletePage(PageId page_id) {
    // Todo:
    // 0.   lock latch
    // 1.   Make sure you call DiskManager::DeallocatePage!
    // 2.   Search the page table for the requested page (P).
    // 2.1  If P does not exist, return true.
    // 2.2  If P exists, but has a non-zero pin-count, return false. Someone is using the page.
    // 3.   Otherwise, P can be deleted. Remove P from the page table, reset its metadata and return it to the free
    // list.

    std::scoped_lock lock{latch_};

    // 1.   在page_table_中查找目标页，若不存在返回true
    if(page_table_.count(page_id) == 0){
        return true;
    }
    frame_id_t fid = page_table_[page_id];
    //将目标页从replacer的队列中删除，防止被其他进程引用
    replacer_->Pin(fid);
    // 2.   若目标页的pin_count不为0，则返回false
    Page *page = &(pages_[fid]);
    if(page->pin_count_ != 0){
        replacer_->Unpin(fid);
        return false;
    }
    // 3.   将目标页数据写回磁盘
    //原本此处调用flush_page()函数，但调用需要临时解锁操作，会带来并发冲突问题
    //为了解决此问题，将flush_page()函数剥离出来
    PageId old_page_id = page_id;
    if(page_table_.count(old_page_id) > 0){
        frame_id_t fid = page_table_[old_page_id];
        Page *page = &(pages_[fid]);
        disk_manager_->write_page(old_page_id.fd, old_page_id.page_no, page->data_,PAGE_SIZE);
        page->is_dirty_ = false;
    }
    
    //4. 从页表中删除目标页，重置其元数据，将其加入free_list_，返回true
    page_table_.erase(page_id);
    page->ResetMemory();
    page->pin_count_ = 0;

    free_list_.emplace_back(fid);

    return true;

/*     std::scoped_lock lock{latch_};

    std::unordered_map<PageId, frame_id_t, PageIdHash>::iterator page_it = page_table_.find(page_id);
    if(page_it == page_table_.end())  return true;

    Page *p = &pages_[page_it->second];
    if(p->pin_count_ != 0)   return false;

    disk_manager_->DeallocatePage(page_id.page_no);
    page_id.page_no = INVALID_PAGE_ID;
    UpdatePage(p, page_id, page_it->second);
    free_list_.push_back(page_it->second);
    return true; */
}

/**
 * @brief Flushes all the pages in the buffer pool to disk.
 *
 * @param fd 指定的diskfile open句柄
 * 
 * 将buffer_pool中的所有页写回磁盘，要参数fd干什么？
 */
void BufferPoolManager::FlushAllPages(int fd) {
    std::scoped_lock lock{latch_};
    for (size_t i = 0; i < pool_size_; i++) {
        Page *page = &pages_[i];
        if (page->GetPageId().fd == fd && page->GetPageId().page_no != INVALID_PAGE_ID) {
            disk_manager_->write_page(page->GetPageId().fd, page->GetPageId().page_no, page->GetData(), PAGE_SIZE);
            page->is_dirty_ = false;
        }
    }
}