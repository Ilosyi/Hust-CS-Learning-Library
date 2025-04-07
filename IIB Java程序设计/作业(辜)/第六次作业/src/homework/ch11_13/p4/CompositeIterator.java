package homework.ch11_13.p4;

import java.util.ArrayList;
import java.util.List;

/**
 * CompositeIterator 是用于遍历组件树的迭代器，采用深度优先遍历。
 * 它持有一个栈结构（用 List 模拟）来保存每一层的 ComponentIterator，
 * 每向下一层，就把当前子组件的迭代器压入栈中，逐层遍历。
 */
public class CompositeIterator implements ComponentIterator {

    // 保存正在遍历的组件迭代器栈结构（模拟DFS）
    protected List<ComponentIterator> iterators;

    /**
     * 构造器，传入树的根节点的迭代器
     * @param iterator 根节点组件的迭代器
     */
    public CompositeIterator(ComponentIterator iterator) {
        iterators = new ArrayList<>();
        iterators.add(iterator);
    }

    /**
     * 判断是否还有下一个组件
     * 实现思路：如果当前栈顶的迭代器还有下一个元素，就返回 true；
     * 否则就弹出栈顶迭代器，继续检查下一层迭代器，直到栈为空或找到有元素的迭代器
     */
    @Override
    public boolean hasNext() {
        while (!iterators.isEmpty()) {
            ComponentIterator currentIterator = iterators.get(iterators.size() - 1); // 查看栈顶
            if (currentIterator.hasNext()) {
                return true;
            } else {
                iterators.remove(iterators.size() - 1); // 当前迭代器用完，出栈
            }
        }
        return false;
    }

    /**
     * 获取下一个组件
     * 实现思路：在栈顶迭代器中获取下一个组件，
     * 如果这个组件是 CompositeComponent（复合组件），
     * 就将它的子组件迭代器压入栈中，继续向下遍历
     */
    @Override
    public Component next() {
        if (!hasNext()) return null;

        ComponentIterator currentIterator = iterators.get(iterators.size() - 1); // 栈顶迭代器
        Component component = currentIterator.next();

        // 如果这个组件是复合组件，就压入它的子组件迭代器
        if (component instanceof CompositeComponent) {
            ComponentIterator childIterator = component.createIterator();
            iterators.add(childIterator); // 向栈中压入新迭代器
        }

        return component;
    }
}
