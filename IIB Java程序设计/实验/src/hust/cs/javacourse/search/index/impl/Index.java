package hust.cs.javacourse.search.index.impl;

import hust.cs.javacourse.search.index.*;

import java.io.*;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Index extends AbstractIndex {
    /**
     * 返回索引的字符串表示
     * @return 索引的字符串表示
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        // 输出 docId 到 docPath 的映射
        sb.append("Document ID to Document Path Mapping:\n");
        sb.append("------------------------------------\n");
        for (int docId : this.docIdToDocPathMapping.keySet()) {
            sb.append(String.format("%-5d ---> %s\n", docId, this.docIdToDocPathMapping.get(docId)));
        }
        sb.append("\nTotal Mappings: ").append(this.docIdToDocPathMapping.size()).append("\n\n");

        // 输出 term 到 postingList 的映射
        sb.append("Term to PostingList Mapping:\n");
        sb.append("----------------------------\n");
        for (AbstractTerm term : this.termToPostingListMapping.keySet()) {
            sb.append(String.format("%-20s ---> %s\n", term.toString(), this.termToPostingListMapping.get(term).toString()));
        }
        sb.append("\nTotal Mappings: ").append(this.termToPostingListMapping.size()).append("\n");

        return sb.toString();
    }

    /**
     * 将文档对象加入索引结构
     * @param document ：一个文档对象
     */
    @Override
    public void addDocument(AbstractDocument document){
        int docId = document.getDocId();         // 获取文档ID
        String docPath = document.getDocPath();  // 获取文档路径
        docIdToDocPathMapping.put(docId, docPath);  // 建立ID与路径的映射

        for (AbstractTermTuple tuple : document.getTuples()) {
            AbstractTerm term = tuple.term;  // 获取词项
            AbstractPostingList postingList = termToPostingListMapping.get(term);  // 获取该词项的PostingList
            if (postingList == null) {
                postingList = new PostingList(); // 如果没有，则创建一个新的
            }
            int docIndex = postingList.indexOf(docId); // 查看该PostingList中是否已有此文档
            AbstractPosting posting;
            if (docIndex != -1) {
                posting = postingList.get(docIndex);  // 已有则取出
            } else {
                posting = new Posting(docId, 0, new ArrayList<>());  // 否则新建一个
                postingList.add(posting);  // 添加新的Posting
            }
            posting.setFreq(posting.getFreq() + 1);  // 更新词频
            posting.getPositions().add(tuple.curPos); // 添加词出现的位置
            termToPostingListMapping.put(term, postingList); // 更新 term 到 PostingList 的映射
        }
    }

    /**
     * 从文件中加载索引结构
     * @param file ：已存储索引的文件
     */
    @Override
    public void load(File file){
        try {
            readObject(new ObjectInputStream(new FileInputStream(file)));
        } catch (Exception e){
            // 使用日志框架记录异常
            Logger.getLogger(Index.class.getName()).log(Level.SEVERE, "Failed to load index from file", e);
        }
    }

    /**
     * 将索引保存到文件中
     * @param file ：目标文件
     */
    @Override
    public void save(File file){
        try {
            writeObject(new ObjectOutputStream(new FileOutputStream(file)));
        } catch (Exception e){
            // 使用日志框架记录异常
            Logger.getLogger(Index.class.getName()).log(Level.SEVERE, "Failed to save index to file", e);
        }
    }

    /**
     * 搜索指定词项对应的PostingList
     * @param term : 查询的词项
     * @return ：返回对应的PostingList（可为null）
     */
    @Override
    public AbstractPostingList search(AbstractTerm term){
        return termToPostingListMapping.get(term);
    }

    /**
     * 获取索引中所有词项的集合（即字典）
     * @return ：Set<AbstractTerm>
     */
    @Override
    public Set<AbstractTerm> getDictionary(){
        return termToPostingListMapping.keySet();
    }

    /**
     * 优化索引：对PostingList和词位位置进行排序
     */
    @Override
    public void optimize(){
        for (AbstractTerm term : getDictionary()) {
            AbstractPostingList postingList = termToPostingListMapping.get(term);
            postingList.sort(); // 对postingList按docId升序
            for (int i = 0; i < postingList.size(); i++) {
                postingList.get(i).sort(); // 对每个posting的positions排序
            }
        }
    }

    /**
     * 根据docId获取文档路径
     * @param docId ：文档ID
     * @return ：完整路径字符串
     */
    @Override
    public String getDocName(int docId){
        return docIdToDocPathMapping.get(docId);
    }

    /**
     * 将索引对象写入到输出流中（序列化）
     * @param out : ObjectOutputStream输出流
     */


    @Override
    public void writeObject(ObjectOutputStream out) {
        try {
            // 1. 将文档ID与路径的映射写出
            out.writeObject(docIdToDocPathMapping);

            // 2. 将词项与其PostingList的映射写出
            out.writeObject(termToPostingListMapping);
        } catch (Exception e) {
            // 使用日志框架记录异常
            Logger.getLogger(Index.class.getName()).log(Level.SEVERE, "Failed to write object to output stream", e);
        }
    }

    /**
     * 从输入流中读取索引对象（反序列化）
     * @param in ：ObjectInputStream输入流
     */

    @Override
    @SuppressWarnings("unchecked")
    public void readObject(ObjectInputStream in) {
        try {
            // 1. 从输入流中读取文档ID到路径的映射
            Object docIdToDocPathMappingObj = in.readObject();
            if (docIdToDocPathMappingObj instanceof Map) {
                docIdToDocPathMapping = (Map<Integer, String>) docIdToDocPathMappingObj;
            } else {
                throw new ClassCastException("Expected Map<Integer, String>, but got " + docIdToDocPathMappingObj.getClass());
            }

            // 2. 读取词项到PostingList的映射
            Object termToPostingListMappingObj = in.readObject();
            if (termToPostingListMappingObj instanceof Map) {
                termToPostingListMapping = (Map<AbstractTerm, AbstractPostingList>) termToPostingListMappingObj;
            } else {
                throw new ClassCastException("Expected Map<AbstractTerm, AbstractPostingList>, but got " + termToPostingListMappingObj.getClass());
            }
        } catch (Exception e) {
            // 使用日志框架记录异常
            Logger.getLogger(Index.class.getName()).log(Level.SEVERE, "Failed to read object from input stream", e);
        }
    }
}